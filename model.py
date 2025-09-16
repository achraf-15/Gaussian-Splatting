import os
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import MS_SSIM
from renderer_wrapper import GaussianRenderer
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm
import numpy as np

from PIL import Image
import torchvision.transforms as T
import torch

# ------------------------------
# Utilities
# ------------------------------
def compute_luminance(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.299*r + 0.587*g + 0.114*b

def sobel_grad_map(img):
    device = img.device
    gray = compute_luminance(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    kx = torch.tensor([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]], device=device).view(1,1,3,3)
    ky = torch.tensor([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]], device=device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12).squeeze()

def sample_pixels_from_prob(prob_map, n_samples, replace=False):
    H, W = prob_map.shape
    flat = prob_map.reshape(-1)
    flat = flat / (flat.sum() + 1e-12)
    idx = torch.multinomial(flat, n_samples, replacement=replace)
    ys, xs = (idx // W).long(), (idx % W).long()
    return xs, ys, idx

# ------------------------------
# Main Optimizer Class
# ------------------------------
class ImageGSOptimizer:
    def __init__(self, target_image, save_dir="output", device='cuda', **kwargs):
        self.device = device
        self.target_image = target_image.to(device)
        self.H, self.W, _ = target_image.shape
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Default parameters (can be overridden)
        self.N_total = kwargs.get('N_total', 1000)
        self.steps = kwargs.get('steps', 1500)
        self.H_t = kwargs.get('H_t', 32)
        self.W_t = kwargs.get('W_t', 32)
        self.K = kwargs.get('K', 10)
        self.max_G_per_tile = kwargs.get('max_G_per_tile', 100)
        self.lambda_init = kwargs.get('lambda_init', 0.3)
        self.init_scale_pixels = kwargs.get('init_scale_pixels', 5.0)
        self.add_interval = kwargs.get('add_interval', 500)
        self.add_count_per_step = kwargs.get('add_count_per_step', None)
        if self.add_count_per_step is None:
            self.add_count_per_step = max(1, self.N_total // 8)

        # Renderer
        self.renderer = GaussianRenderer(self.H, self.W, H_t=self.H_t, W_t=self.W_t,
                                         K=self.K, max_G_per_tile=self.max_G_per_tile)

        # Losses
        self.ms_ssim_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.l1 = torch.nn.L1Loss()

    # ------------------------------
    # Gaussian Initialization
    # ------------------------------
    def initialize_gaussians(self):
        # Content-adaptive probabilities (Eq.6)
        gmag = sobel_grad_map(self.target_image)
        g_flat_sum = gmag.sum().item()
        if g_flat_sum == 0:
            gprob = torch.zeros_like(gmag)
        else:
            gprob = gmag / g_flat_sum
        uniform = torch.full_like(gprob, 1.0/(self.H*self.W))
        P_init = (1.0 - self.lambda_init) * gprob + self.lambda_init * uniform

        # Initial Gaussian positions
        N_init = self.N_total // 2
        xs, ys, _ = sample_pixels_from_prob(P_init, N_init, replace=True)

        means = torch.stack([xs.float(), ys.float()], dim=1).to(self.device)
        rotations = torch.zeros(N_init, device=self.device)
        # log_scales = torch.full((N_init,2), math.log(self.init_scale_pixels), device=self.device)
        inv_scales = torch.full((N_init,2), 1.0/self.init_scale_pixels, device=self.device)
        colors = self.target_image[ys, xs, :].to(self.device).float()

        means.requires_grad_(True)
        rotations.requires_grad_(True)
        inv_scales.requires_grad_(True)
        colors.requires_grad_(True)

        return means, rotations, inv_scales, colors

    # ------------------------------
    # Optimizer Setup
    # ------------------------------
    def make_optimizer(self, means, rotations, inv_scales, colors):
        return optim.Adam([
            {"params":[means], "lr":5e-4},
            {"params":[colors], "lr":5e-3},
            {"params":[inv_scales], "lr":2e-3},
            {"params":[rotations], "lr":2e-3},
        ])

    # ------------------------------
    # Training Loop
    # ------------------------------
    def optimize(self, visualize_interval=500, plot_last=True):
        # Initialization
        means, rotations, inv_scales, colors = self.initialize_gaussians()
        optimizer = self.make_optimizer(means, rotations, inv_scales, colors)
        N_current = means.shape[0]
        add_steps = set(range(self.add_interval, self.steps+1, self.add_interval))
        saved_debug = {}

        self.visualize('Initialization', means, inv_scales, rotations, colors, rendered=None)


        for step in tqdm(range(1, self.steps+1), desc="Rendering Image"):
            optimizer.zero_grad()
            rendered = self.renderer(means, rotations, inv_scales, colors)
            loss_l1 = self.l1(rendered, self.target_image)
            loss_mssim = 1.0 - self.ms_ssim_fn(rendered.permute(2,0,1).unsqueeze(0),
                                              self.target_image.permute(2,0,1).unsqueeze(0))
            loss = loss_l1 + 0.1*loss_mssim
            loss.backward()
            optimizer.step()

            # Clamp physically valid ranges
            with torch.no_grad():
                # inv_scales should be positive (paper uses 1/s). Clamp to avoid sign flips.
                inv_scales.clamp_(min=1e-4, max=1.0)  # min corresponds to large s, max to s=1
                # rotations in [0, pi]
                rotations[:] = torch.remainder(rotations, math.pi)
                 # colors in [0,1]
                colors.clamp_(0.0,1.0)

            # Progressive addition
            if step in add_steps and N_current < self.N_total:
                with torch.no_grad():
                    err = torch.abs(rendered - self.target_image).sum(dim=2)
                    total_err = err.sum()
                    P_add = err/(total_err + 1e-12) if total_err.item()>0 else torch.full_like(err, 1.0/(self.H*self.W))
                    n_add = min(self.add_count_per_step, self.N_total - N_current)
                    xs_add, ys_add, _ = sample_pixels_from_prob(P_add, n_add, replace=False)

                    new_means = torch.stack([xs_add.float(), ys_add.float()], dim=1).to(self.device)
                    new_rot = torch.zeros(n_add, device=self.device)
                    # new_log_sc = torch.full((n_add,2), math.log(self.init_scale_pixels), device=self.device)
                    new_log_sc = torch.full((n_add,2), 1.0/self.init_scale_pixels, device=self.device)
                    new_colors = self.target_image[ys_add, xs_add, :].to(self.device).float()

                    # Concatenate to params
                    means = torch.cat([means.detach(), new_means], dim=0).requires_grad_(True)
                    rotations = torch.cat([rotations.detach(), new_rot], dim=0).requires_grad_(True)
                    inv_scales = torch.cat([inv_scales.detach(), new_log_sc], dim=0).requires_grad_(True)
                    colors = torch.cat([colors.detach(), new_colors], dim=0).requires_grad_(True)
                    N_current = means.shape[0]
                    optimizer = self.make_optimizer(means, rotations, inv_scales, colors)

            # Save debug info
            if step % visualize_interval == 0 or step == 1 or step == self.steps:
                saved_debug[step] = {"loss": loss.item(), "N": N_current, "rendered": rendered.detach().cpu()}
                self.visualize(step, means, inv_scales, rotations, colors, rendered, plot=False)

        self.visualize('Final', means, inv_scales, rotations, colors, rendered, plot=True)

        return means, rotations, inv_scales, colors, saved_debug
    
    def visualize(self, step, means, inv_scales, rotations, colors, rendered=None, plot=False):
        # Set figsize and DPI so that HxW pixels exactly
        dpi = 100
        if rendered is not None: 
            fig, axes = plt.subplots(1, 3, figsize=(3*self.W/dpi, self.H/dpi), dpi=dpi)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(2*self.W/dpi, self.H/dpi), dpi=dpi)

        # Plot target image
        img_to_show = torch.clamp(self.target_image.detach().cpu(), 0.0, 1.0)
        axes[0].imshow(img_to_show)
        axes[0].set_title('Target Image')
        axes[0].axis('off')
        
        axes[1].set_facecolor('black')
        #scales = torch.exp(inv_scales).detach().cpu().numpy()
        scales = 1.0 / inv_scales.detach().cpu().numpy()
        means_np = means.detach().cpu().numpy()
        rotations_np = rotations.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()
        
        for (x, y), (sx, sy), theta, color in zip(means_np, scales, rotations_np, colors_np):
            ell = Ellipse((x, y), width=6*sx, height=6*sy,
                        angle=theta*180/math.pi, edgecolor=color, facecolor=color, alpha=0.5)
            axes[1].add_patch(ell)

        axes[1].set_xlim(0, self.W)
        axes[1].set_ylim(self.H, 0)
        axes[1].set_title('Gaussian Blobs - Step:'+str(step))
        axes[1].axis('off')

        fig.canvas.draw()

        # Plot rendered image
        if rendered is not None:
            img_to_show = torch.clamp(rendered.detach().cpu(), 0.0, 1.0)
            axes[2].imshow(img_to_show)
            axes[2].set_title('Rendered Image - Step:'+str(step))
            axes[2].axis('off')

        if plot:
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_dir,'visualization_step_'+str(step)+'.png')) 
            plt.close(fig)

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":

    device='cuda'

    # Load your image
    img_path = "./kodak/kodim17.png"  # replace with your file path
    img_pil = Image.open(img_path).convert("RGB")  # ensure 3 channels

    # Transform to tensor in [0,1] and send to device
    transform = T.ToTensor()  # converts to [C,H,W] float in [0,1]
    target_image = transform(img_pil).permute(1, 2, 0).to(device)  # [H,W,3]

    optimizer = ImageGSOptimizer(target_image, save_dir="gs_output", device=device,
                                 N_total=10000, steps=1500, H_t=16, W_t=16, add_interval=250)
    means, rotations, inv_scales, colors, debug = optimizer.optimize()
