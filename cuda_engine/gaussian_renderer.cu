// gaussian_renderer.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

#include "sorting.cuh"
#include "utils.cuh"

#define MAX_G_PER_TILE 128  // compile-time upper bound, must be >= max_G_per_tile used from Python


// CUDA kernel for tile-Gaussian correspondence
__global__ void findTileGaussianCorrespondence(
    const float* gaussian_means,    // [N_g, 2]
    const float* gaussian_rotations, // [N_g] (radians)
    const float* gaussian_inv_scales, // [N_g, 2] (inv of scales)
    int N_g,                         // Number of Gaussians
    int H_t,                          // Tile height
    int W_t,                          // Tile width
    int H,                            // Image height
    int W,                            // Image width
    int* tile_gaussian_indices,      // [N_t, max_G_per_tile]
    int* tile_gaussian_counts,       // [N_t]
    int max_G_per_tile
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int n_tiles_x = (W + W_t - 1) / W_t;
    int n_tiles_y = (H + H_t - 1) / H_t;

    if (tx >= n_tiles_x || ty >= n_tiles_y) return;

    int tile_idx = ty * n_tiles_x + tx;

    float tile_x_min = tx * W_t;
    float tile_x_max = fminf((tx+1)*W_t, (float)W);
    float tile_y_min = ty * H_t;
    float tile_y_max = fminf((ty+1)*H_t, (float)H);

    int count = 0;
    for (int g = 0; g < N_g; g++) {
        // Get Gaussian parameters
        float g_x = gaussian_means[g * 2 + 0];
        float g_y = gaussian_means[g * 2 + 1];

        float inv_sx = gaussian_inv_scales[g * 2 + 0];
        float inv_sy = gaussian_inv_scales[g * 2 + 1];
        float sx =1.0f / inv_sx;
        float sy =1.0f / inv_sy;

        float radius = 3.f * fmaxf(sx, sy);

        // Check intersection with tile
        bool intersects =
            (g_x + radius >= tile_x_min) && (g_x - radius <= tile_x_max) &&
            (g_y + radius >= tile_y_min) && (g_y - radius <= tile_y_max);

        if (intersects && count < max_G_per_tile) {
            tile_gaussian_indices[tile_idx * max_G_per_tile + count] = g;
            count++;
        }
    }
    tile_gaussian_counts[tile_idx] = count;
    tile_gaussian_counts[tile_idx] = min(count, max_G_per_tile);

}


// CUDA kernel for rendering
__global__ void renderTile(
    const float* gaussian_means,    // [N_g, 2]
    const float* gaussian_rotations, // [N_g] (radians)
    const float* gaussian_inv_scales, // [N_g, 2] (inv of scales)
    const float* gaussian_colors,   // [N_g, 3]
    const int* tile_gaussian_indices,// [N_t, max_G_per_tile]
    const int* tile_gaussian_counts, // [N_t]
    int* pixel_topk_indices,
    int N_g,                         // Number of Gaussians
    int K,                           // Top-K
    int H_t,                         // Tile height
    int W_t,                         // Tile width
    int H,                           // Image height
    int W,                           // Image width
    float* output_image,            // [H, W, 3]
    int max_G_per_tile
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    int n_tiles_x = (W + W_t - 1) / W_t;
    int tile_x = x / W_t;
    int tile_y = y / H_t;
    int tile_idx = tile_y * n_tiles_x + tile_x;

    int count = tile_gaussian_counts[tile_idx];
    count = min(count, MAX_G_PER_TILE);
    if (count == 0) return;

    const int* g_indices = &tile_gaussian_indices[tile_idx * max_G_per_tile];

    extern __shared__ float sdata[];
    float* s_mu    = sdata;                                 // 2*max_G_per_tile
    float* s_theta = s_mu + 2*max_G_per_tile;               // 1*max
    float* s_inv   = s_theta + max_G_per_tile;             // 2*max
    float* s_col   = s_inv + 2*max_G_per_tile;             // 3*max
    

    // Cooperative load of tile Gaussians into shared memory; Each thread in the block participates in loading the count Gaussians into shared memory 
    int blockThreads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < count; i += blockThreads) {
        int g = g_indices[i];
        // defensive: invalid g -> write zeros
        if (g < 0 || g >= N_g) {
            printf("renderTile: BAD_G_INDEX g=%d at tile_idx=%d pixel=(%d,%d) count=%d N_g=%d n_tiles_x=%d\\n",
                   g, tile_idx, x, y, count, N_g, n_tiles_x);
            s_mu[i*2 + 0] = 0.0f;
            s_mu[i*2 + 1] = 0.0f;
            s_theta[i]    = 0.0f;
            s_inv[i*2 + 0]= 0.0f;
            s_inv[i*2 + 1]= 0.0f;
            s_col[i*3 + 0]= 0.0f;
            s_col[i*3 + 1]= 0.0f;
            s_col[i*3 + 2]= 0.0f;
        } else {
            // load mean & rotation & inv-scales 
            float mu_x = gaussian_means[g*2 + 0];
            float mu_y = gaussian_means[g*2 + 1];
            float theta = gaussian_rotations[g];
            float inv_sx = gaussian_inv_scales[g*2 + 0];
            float inv_sy = gaussian_inv_scales[g*2 + 1];

            s_mu[i*2 + 0] = mu_x;
            s_mu[i*2 + 1] = mu_y;
            s_theta[i]    = theta;
            s_inv[i*2 + 0]= inv_sx;
            s_inv[i*2 + 1]= inv_sy;
            s_col[i*3 + 0]= gaussian_colors[g*3 + 0];
            s_col[i*3 + 1]= gaussian_colors[g*3 + 1];
            s_col[i*3 + 2]= gaussian_colors[g*3 + 2];
        }
    }
    __syncthreads(); // all shared data loaded


    float g_values[MAX_G_PER_TILE];          
    int topK_indices[MAX_G_PER_TILE]; 
    
    // Evaluate Gaussians in this tile for this pixel 
    for (int i = 0; i < count; ++i) {
        // read parameters from shared memory (cheap)
        float mu_x = s_mu[i*2 + 0];
        float mu_y = s_mu[i*2 + 1];
        float theta = s_theta[i];
        float inv_sx = s_inv[i*2 + 0];
        float inv_sy = s_inv[i*2 + 1];

        float val = gaussian_eval((float)x, (float)y, mu_x, mu_y, theta, inv_sx, inv_sy);
        g_values[i] = val;

        topK_indices[i] = i;
    }

    topk_selector(g_values, topK_indices, count, K); 
    

    // Write top-K **local indices** to global memory
    int useK = min(K, count);
    int* pixel_topk_ptr = &pixel_topk_indices[(y*W + x)*K];
    for (int i = 0; i < useK; i++) {
        int local_idx = topK_indices[i];   // index into gs array
        if (local_idx < 0 || local_idx >= count) {
            printf("renderTile: BAD_LOCAL_INDEX local_idx=%d at tile_idx=%d pixel=(%d,%d) count=%d\\n",
                local_idx, tile_idx, x, y, count);
            pixel_topk_ptr[i] = -1; // mark invalid
            continue;
        }
        pixel_topk_ptr[i] = topK_indices[i]; // store local index, NOT global
    }
 
    // Aggregate colors
    float sum_weights = 0.0f;
    float sum_colors[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < useK; i++) {
        int idx = topK_indices[i];
        float val = g_values[idx];
        sum_weights += val;
        sum_colors[0] += val * s_col[idx*3+0];
        sum_colors[1] += val * s_col[idx*3+1];
        sum_colors[2] += val * s_col[idx*3+2];
    }

    if (sum_weights > 1e-8f) {
        output_image[(y * W + x) * 3 + 0] = sum_colors[0] / sum_weights;
        output_image[(y * W + x) * 3 + 1] = sum_colors[1] / sum_weights;
        output_image[(y * W + x) * 3 + 2] = sum_colors[2] / sum_weights;
    } else {
        // black or previous value (you may prefer to write 0)
        output_image[(y * W + x) * 3 + 0] = 0.0f;
        output_image[(y * W + x) * 3 + 1] = 0.0f;
        output_image[(y * W + x) * 3 + 2] = 0.0f;
    }
}

// CUDA kernels for backward pass
__global__ void renderTileBackward(
    const float* grad_output,        // Gradient of the output image [H, W, 3]
    const float* gaussian_means,    // [N_g, 2]
    const float* gaussian_rotations, // [N_g] (radians)
    const float* gaussian_inv_scales, // [N_g, 2] (inv of scales)
    const float* gaussian_colors,   // [N_g, 3]
    const int* tile_gaussian_indices,// [N_t, max_G_per_tile]
    const int* tile_gaussian_counts, // [N_t]
    float* grad_means,               // Gradient of means [N_g, 2]
    float* grad_rotations,           // Gradient of rotations [N_g]
    float* grad_inv_scales,          // Gradient of inverse scales [N_g, 2]
    float* grad_colors,              // Gradient of colors [N_g, 3]
    const int* pixel_topk_indices,
    int N_g,                         // Number of Gaussians
    int K,                           // Top-K
    int H_t,                         // Tile height
    int W_t,                         // Tile width
    int H,                           // Image height
    int W,                           // Image width
    int max_G_per_tile
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    int n_tiles_x = (W + W_t - 1) / W_t;
    int tile_x = x / W_t;
    int tile_y = y / H_t;
    int tile_idx = tile_y * n_tiles_x + tile_x;

    int count = tile_gaussian_counts[tile_idx];
    count = min(count, MAX_G_PER_TILE);
    if (count == 0) return;

    const int* g_indices = &tile_gaussian_indices[tile_idx * max_G_per_tile];

    extern __shared__ char s_mem[]; // allocated at launch in bytes
    float* s_f = (float*) s_mem; // pointer to float area
    float* s_mu    = s_f;                                        // 2 * max
    float* s_theta = s_mu    + 2 * max_G_per_tile;               // 1 * max
    float* s_inv   = s_theta +    max_G_per_tile;               // 2 * max
    float* s_col   = s_inv    + 2 * max_G_per_tile;             // 3 * max
    int*   s_idx   = (int*)(s_col + 3 * max_G_per_tile);        // int area (max entries)

    // Cooperative load tile parameters into shared memory
    int blockThreads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < count; i += blockThreads) {
        int g = g_indices[i];
        if (g < 0 || g >= N_g) {
            // invalid entry -> zero-out & mark
            s_mu[i*2 + 0] = 0.0f;
            s_mu[i*2 + 1] = 0.0f;
            s_theta[i] = 0.0f;
            s_inv[i*2 + 0] = 0.0f;
            s_inv[i*2 + 1] = 0.0f;
            s_col[i*3 + 0] = 0.0f;
            s_col[i*3 + 1] = 0.0f;
            s_col[i*3 + 2] = 0.0f;
            s_idx[i] = -1;
        } else {
            float mu_x = gaussian_means[g*2 + 0];
            float mu_y = gaussian_means[g*2 + 1];
            float theta = gaussian_rotations[g];
            float inv_sx = gaussian_inv_scales[g*2 + 0];
            float inv_sy = gaussian_inv_scales[g*2 + 1];

            s_mu[i*2 + 0] = mu_x;
            s_mu[i*2 + 1] = mu_y;
            s_theta[i]    = theta;
            s_inv[i*2 + 0]= inv_sx;
            s_inv[i*2 + 1]= inv_sy;
            s_col[i*3 + 0] = gaussian_colors[g*3 + 0];
            s_col[i*3 + 1] = gaussian_colors[g*3 + 1];
            s_col[i*3 + 2] = gaussian_colors[g*3 + 2];
            s_idx[i] = g;
        }
    }
    __syncthreads(); // now shared arrays ready for this tile/block

    // Per-thread per-pixel values (only these are per-thread)
    float g_values[MAX_G_PER_TILE];

    // Evaluate all gaussians for this pixel using shared parameters
    for (int i = 0; i < count; ++i) {
        float mu_x = s_mu[i*2 + 0];
        float mu_y = s_mu[i*2 + 1];
        float theta = s_theta[i];
        float inv_sx = s_inv[i*2 + 0];
        float inv_sy = s_inv[i*2 + 1];
        g_values[i] = gaussian_eval((float)x, (float)y, mu_x, mu_y, theta, inv_sx, inv_sy);
    }

    // read top-K local indices computed by forward (indices into tile list)
    int useK = min(K,count);
    const int* topK_indices = &pixel_topk_indices[(y*W + x)*K];

    // aggregate for pixel color reconstruction (match forward)
    float sum_weights = 0.0f;
    float sum_colors[3] = {0.0f,0.0f,0.0f};

    for (int i = 0; i < useK; i++) {
        int g_local  = topK_indices[i];
        if (g_local  < 0 || g_local  >= N_g) {
            printf("renderTileBackward 2: BAD_G_INDEX g=%d at tile_idx=%d pixel=(%d,%d) count=%d N_g=%d n_tiles_x=%d\\n",
                   g_local , tile_idx, x, y, count, N_g, n_tiles_x);
            continue;
        }

        float w = g_values[g_local];
        sum_weights += w;
        sum_colors[0] += w * s_col[g_local*3 + 0];
        sum_colors[1] += w * s_col[g_local*3 + 1];
        sum_colors[2] += w * s_col[g_local*3 + 2];
    }
    if (sum_weights < 1e-8f) return;

    // Reconstructed pixel color (matching forward)
    float I_c[3] = { sum_colors[0]/sum_weights, sum_colors[1]/sum_weights, sum_colors[2]/sum_weights };
    float go[3] = {grad_output[(y * W + x)*3 + 0], grad_output[(y * W + x)*3 + 1], grad_output[(y * W + x)*3 + 2]};

    // compute grads (use shared params + per-thread g_values)
    for(int i=0;i<useK;i++){
        int g_local = topK_indices[i];
        if (g_local < 0 || g_local >= count) continue;

        //int g = gs[g_local].idx;            // global Gaussian ID for writing gradients
        int g = s_idx[g_local];           // global gaussian id
        if (g < 0 || g >= N_g) continue;

        float Gk = g_values[g_local];

        // Color gradient: dL/dc_k = (G_k / S) * go[c]
        float weight = g_values[g_local] / sum_weights;
        for(int c=0; c<3; c++){
            float add = weight * go[c];
            atomicAdd(&grad_colors[g*3 + c], add);
        }

        // Gradient wrt G_k:
        // dL/dG_k = sum_c go[c] * (c_k[c] - I_c[c]) / S
        float dL_dGk = 0.0f;
        for (int c=0; c<3; c++){
            dL_dGk += go[c] * (s_col[g_local*3 + c] - I_c[c]);
        }
        dL_dGk /= sum_weights;

        // propagate through gaussian eval:
        // read shared params for this local index
        float mu_x = s_mu[g_local*2 + 0];
        float mu_y = s_mu[g_local*2 + 1];
        float theta = s_theta[g_local];
        float inv_sx = s_inv[g_local*2 + 0];
        float inv_sy = s_inv[g_local*2 + 1];

        // call gaussian_gradients with g_val = Gk (so returned derivatives already multiply by Gk)
        float dmu_x, dmu_y, dtheta, dinv_sx, dinv_sy;
        gaussian_gradients((float)x, (float)y, mu_x, mu_y, theta, inv_sx, inv_sy, Gk,
                           dmu_x, dmu_y, dtheta, dinv_sx, dinv_sy);

        // Apply gradient
        atomicAdd(&grad_means[g*2+0], dmu_x * dL_dGk);
        atomicAdd(&grad_means[g*2+1], dmu_y * dL_dGk);
        atomicAdd(&grad_rotations[g], dtheta * dL_dGk);
        atomicAdd(&grad_inv_scales[g*2+0], dinv_sx  * dL_dGk);
        atomicAdd(&grad_inv_scales[g*2+1], dinv_sy * dL_dGk);
    }
}


// ============================================================
// Wrappers
// ============================================================

// Wrapper functions for PyTorch
void find_tile_gaussian_correspondence_wrapper(
    torch::Tensor gaussian_means,
    torch::Tensor gaussian_rotations,
    torch::Tensor gaussian_inv_scales,
    int N_g,
    int H_t,
    int W_t,
    int H,
    int W,
    torch::Tensor tile_gaussian_indices,
    torch::Tensor tile_gaussian_counts,
    int max_G_per_tile
) {
    const float* gaussian_means_ptr = gaussian_means.data_ptr<float>();
    const float* gaussian_rotations_ptr = gaussian_rotations.data_ptr<float>();
    const float* gaussian_inv_scales_ptr = gaussian_inv_scales.data_ptr<float>();
    int* tile_gaussian_indices_ptr = tile_gaussian_indices.data_ptr<int>();
    int* tile_gaussian_counts_ptr = tile_gaussian_counts.data_ptr<int>();

    // Compute tile counts
    int n_tiles_x = (W + W_t - 1) / W_t;
    int n_tiles_y = (H + H_t - 1) / H_t;
    int N_t = n_tiles_x * n_tiles_y;

    // Initialize device memory: set counts to 0 and indices to -1
    cudaMemset(tile_gaussian_counts_ptr, 0, sizeof(int) * (size_t)N_t);
    // Set indices to -1 (0xFF bytes)
    cudaMemset(tile_gaussian_indices_ptr, 0xFF, sizeof(int) * (size_t)N_t * (size_t)max_G_per_tile);

    dim3 block(16, 16);
    dim3 grid((n_tiles_x + block.x - 1) / block.x, (n_tiles_y + block.y - 1) / block.y);

    findTileGaussianCorrespondence<<<grid, block>>>(
        gaussian_means_ptr,
        gaussian_rotations_ptr,
        gaussian_inv_scales_ptr,
        N_g,
        H_t,
        W_t,
        H,
        W,
        tile_gaussian_indices_ptr,
        tile_gaussian_counts_ptr,
        max_G_per_tile
    );
    //cudaDeviceSynchronize();
}

void render_tile_wrapper(
    torch::Tensor gaussian_means,
    torch::Tensor gaussian_rotations,
    torch::Tensor gaussian_inv_scales,
    torch::Tensor gaussian_colors,
    torch::Tensor tile_gaussian_indices,
    torch::Tensor tile_gaussian_counts,
    torch::Tensor output_image,
    torch::Tensor pixel_topk_indices,
    int N_g,                        
    int K,
    int H_t,
    int W_t,
    int H,
    int W,
    int max_G_per_tile
) {
    const float* gaussian_means_ptr = gaussian_means.data_ptr<float>();
    const float* gaussian_rotations_ptr = gaussian_rotations.data_ptr<float>();
    const float* gaussian_inv_scales_ptr = gaussian_inv_scales.data_ptr<float>();
    const float* gaussian_colors_ptr = gaussian_colors.data_ptr<float>();
    const int* tile_gaussian_indices_ptr = tile_gaussian_indices.data_ptr<int>();
    const int* tile_gaussian_counts_ptr = tile_gaussian_counts.data_ptr<int>();
    float* output_image_ptr = output_image.data_ptr<float>();
    int* pixel_topk_indices_ptr = pixel_topk_indices.data_ptr<int>();

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    
    // Shared memory: 2+1+2+3 floats per Gaussian = 8 floats per Gaussian
    size_t shared_bytes = sizeof(float) * 8 * max_G_per_tile;

    renderTile<<<grid, block, shared_bytes>>>(
        gaussian_means_ptr,
        gaussian_rotations_ptr,
        gaussian_inv_scales_ptr,
        gaussian_colors_ptr,
        tile_gaussian_indices_ptr,
        tile_gaussian_counts_ptr,
        pixel_topk_indices_ptr,
        N_g,
        K,
        H_t,
        W_t,
        H,
        W,
        output_image_ptr,
        max_G_per_tile
    );
    //cudaDeviceSynchronize();
}

// Wrapper for backward pass
void render_tile_backward_wrapper(
    torch::Tensor grad_output,
    torch::Tensor gaussian_means,
    torch::Tensor gaussian_rotations,
    torch::Tensor gaussian_inv_scales,
    torch::Tensor gaussian_colors,
    torch::Tensor tile_gaussian_indices,
    torch::Tensor tile_gaussian_counts,
    torch::Tensor grad_means,
    torch::Tensor grad_rotations,
    torch::Tensor grad_inv_scales,
    torch::Tensor grad_colors,
    torch::Tensor pixel_topk_indices,
    int N_g,
    int K,
    int H_t,
    int W_t,
    int H,
    int W,
    int max_G_per_tile
) {
    const float* grad_output_ptr = grad_output.data_ptr<float>();
    const float* gaussian_means_ptr = gaussian_means.data_ptr<float>();
    const float* gaussian_rotations_ptr = gaussian_rotations.data_ptr<float>();
    const float* gaussian_inv_scales_ptr = gaussian_inv_scales.data_ptr<float>();
    const float* gaussian_colors_ptr = gaussian_colors.data_ptr<float>();
    const int* tile_gaussian_indices_ptr = tile_gaussian_indices.data_ptr<int>();
    const int* tile_gaussian_counts_ptr = tile_gaussian_counts.data_ptr<int>();
    float* grad_means_ptr = grad_means.data_ptr<float>();
    float* grad_rotations_ptr = grad_rotations.data_ptr<float>();
    float* grad_inv_scales_ptr = grad_inv_scales.data_ptr<float>();
    float* grad_colors_ptr = grad_colors.data_ptr<float>();
    const int* pixel_topk_indices_ptr = pixel_topk_indices.data_ptr<int>();

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    // compute shared memory
    size_t float_count = (size_t)8 * (size_t)max_G_per_tile; // 8 floats per gaussian (2 mu + 1 theta + 2 inv + 3 color)
    size_t floats_bytes = float_count * sizeof(float);
    size_t ints_bytes = (size_t)max_G_per_tile * sizeof(int);
    size_t shared_bytes = floats_bytes + ints_bytes;

    renderTileBackward<<<grid, block, shared_bytes>>>(
        grad_output_ptr,
        gaussian_means_ptr,
        gaussian_rotations_ptr,
        gaussian_inv_scales_ptr,
        gaussian_colors_ptr,
        tile_gaussian_indices_ptr,
        tile_gaussian_counts_ptr,
        grad_means_ptr,
        grad_rotations_ptr,
        grad_inv_scales_ptr,
        grad_colors_ptr,
        pixel_topk_indices_ptr,
        N_g,
        K,
        H_t,
        W_t,
        H,
        W,
        max_G_per_tile
    );
    //cudaDeviceSynchronize();
}

// Update PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_tile_gaussian_correspondence", &find_tile_gaussian_correspondence_wrapper, "Find tile-Gaussian correspondence");
    m.def("render_tile", &render_tile_wrapper, "Render tile");
    m.def("render_tile_backward", &render_tile_backward_wrapper, "Render tile backward");
}