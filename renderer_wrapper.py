import torch
import torch.nn as nn
from torch.autograd import Function
import cuda_engine.gaussian_renderer_cuda as renderer  # type: ignore

class GaussianRendererFunction(Function):
    @staticmethod
    def forward(ctx, gaussian_means, gaussian_rotations, gaussian_log_scales, gaussian_colors, H, W, H_t, W_t, K, max_G_per_tile):
        N_g = gaussian_means.shape[0]
        N_t = ((H + H_t - 1) // H_t) * ((W + W_t - 1) // W_t)

        # Allocate buffers
        tile_gaussian_indices = torch.zeros(N_t, max_G_per_tile, dtype=torch.int32, device=gaussian_means.device)
        tile_gaussian_counts = torch.zeros(N_t, dtype=torch.int32, device=gaussian_means.device)
        output_image = torch.zeros(H, W, 3, device=gaussian_means.device)
        pixel_topk_indices = torch.full((H*W*K,), -1, dtype=torch.int32, device=gaussian_means.device)

        # Call CUDA kernels
        renderer.find_tile_gaussian_correspondence(
            gaussian_means.contiguous(),
            gaussian_rotations.contiguous(),
            gaussian_log_scales.contiguous(),
            N_g,
            H_t,
            W_t,
            H,
            W,
            tile_gaussian_indices,
            tile_gaussian_counts,
            max_G_per_tile
        )

        renderer.render_tile(
            gaussian_means.contiguous(),
            gaussian_rotations.contiguous(),
            gaussian_log_scales.contiguous(),
            gaussian_colors.contiguous(),
            tile_gaussian_indices,
            tile_gaussian_counts,
            output_image,
            pixel_topk_indices,
            N_g,
            K,
            H_t,
            W_t,
            H,
            W,
            max_G_per_tile
        )

        # Save tensors for backward pass
        ctx.save_for_backward(gaussian_means, gaussian_rotations, gaussian_log_scales, gaussian_colors, tile_gaussian_indices, tile_gaussian_counts, pixel_topk_indices)
        ctx.H, ctx.W, ctx.H_t, ctx.W_t, ctx.K, ctx.max_G_per_tile = H, W, H_t, W_t, K, max_G_per_tile

        return output_image

    @staticmethod
    def backward(ctx, grad_output):
        gaussian_means, gaussian_rotations, gaussian_log_scales, gaussian_colors, tile_gaussian_indices, tile_gaussian_counts, pixel_topk_indices = ctx.saved_tensors
        H, W, H_t, W_t, K, max_G_per_tile = ctx.H, ctx.W, ctx.H_t, ctx.W_t, ctx.K, ctx.max_G_per_tile
        N_g = gaussian_means.shape[0]

        # Allocate buffers for gradients
        grad_means = torch.zeros_like(gaussian_means)
        grad_rotations = torch.zeros_like(gaussian_rotations)
        grad_log_scales = torch.zeros_like(gaussian_log_scales)
        grad_colors = torch.zeros_like(gaussian_colors)

        # Call CUDA kernels for backward pass
        renderer.render_tile_backward(
            grad_output.contiguous(),
            gaussian_means.contiguous(),
            gaussian_rotations.contiguous(),
            gaussian_log_scales.contiguous(),
            gaussian_colors.contiguous(),
            tile_gaussian_indices,
            tile_gaussian_counts,
            grad_means,
            grad_rotations,
            grad_log_scales,
            grad_colors,
            pixel_topk_indices,
            N_g,
            K,
            H_t,
            W_t,
            H,
            W,
            max_G_per_tile
        )

        return grad_means, grad_rotations, grad_log_scales, grad_colors, None, None, None, None, None, None

class GaussianRenderer(nn.Module):
    def __init__(self, H, W, H_t=16, W_t=16, K=10, max_G_per_tile=100):
        super().__init__()
        self.H = H
        self.W = W
        self.H_t = H_t
        self.W_t = W_t
        self.K = K
        self.max_G_per_tile = max_G_per_tile

    def forward(self, gaussian_means, gaussian_rotations, gaussian_log_scales, gaussian_colors):
        return GaussianRendererFunction.apply(
            gaussian_means, gaussian_rotations, gaussian_log_scales, gaussian_colors,
            self.H, self.W, self.H_t, self.W_t, self.K, self.max_G_per_tile
        )
