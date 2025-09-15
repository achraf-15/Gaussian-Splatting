// gaussian_renderer.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

#include "sorting.cuh"



#define MAX_G_PER_TILE 128  // compile-time upper bound, must be >= max_G_per_tile used from Python
#define MAX_K 100


// ============================================================
// Utility: Gaussian evaluation
// ============================================================

__device__ inline float gaussian_eval(
    float x, float y,
    float mu_x, float mu_y,
    float theta, float inv_sx, float inv_sy
) {
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    float inv_var_x = inv_sx * inv_sx;
    float inv_var_y = inv_sy * inv_sy;

    // R = [[cos, -sin], [sin, cos]]
    float R00 = cos_t, R01 = -sin_t;
    float R10 = sin_t, R11 = cos_t;

    // Σ⁻¹ = R diag(inv_var_x, inv_var_y) R^T
    float m00 = R00 * inv_var_x * R00 + R01 * inv_var_y * R01;
    float m01 = R00 * inv_var_x * R10 + R01 * inv_var_y * R11;
    float m11 = R10 * inv_var_x * R10 + R11 * inv_var_y * R11;

    float dx = x - mu_x;
    float dy = y - mu_y;

    float qf = m00 * dx * dx + 2.f * m01 * dx * dy + m11 * dy * dy;

    // Clamp exponent for numerical stability
    float exponent = -0.5f * fminf(qf, 1e4f);
    return expf(exponent);
}

// ============================================================
// Utility: Gaussian gradients
// ============================================================

__device__ inline void gaussian_gradients(
    float x, float y,
    float mu_x, float mu_y,
    float theta, float inv_sx, float inv_sy,
    float g_val,
    float &dmu_x, float &dmu_y,
    float &dtheta, float &dinv_sx, float &dinv_sy
) {
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    float inv_var_x = inv_sx * inv_sx;
    float inv_var_y = inv_sy * inv_sy;

    float R00 = cos_t, R01 = -sin_t;
    float R10 = sin_t, R11 = cos_t;

    // Σ⁻¹
    float m00 = R00 * inv_var_x * R00 + R01 * inv_var_y * R01;
    float m01 = R00 * inv_var_x * R10 + R01 * inv_var_y * R11;
    float m11 = R10 * inv_var_x * R10 + R11 * inv_var_y * R11;

    float dx = x - mu_x;
    float dy = y - mu_y;

    // ∂g/∂μ = -g * M (p-μ)
    float tmp_x = m00 * dx + m01 * dy;
    float tmp_y = m01 * dx + m11 * dy;
    dmu_x = -g_val * tmp_x;
    dmu_y = -g_val * tmp_y;

    // ∂g/∂M = -0.5 g (p-μ)(p-μ)^T
    float outer00 = dx * dx;
    float outer01 = dx * dy;
    float outer11 = dy * dy;

    float dM00 = -0.5f * g_val * outer00;
    float dM01 = -0.5f * g_val * outer01;
    float dM11 = -0.5f * g_val * outer11;

    // Chain rule
    // ∂M/∂inv_var_x = R[:,0]R[:,0]^T
    float dM00_invx = R00 * R00;
    float dM01_invx = R00 * R10;
    float dM11_invx = R10 * R10;

    // ∂M/∂inv_var_y = R[:,1]R[:,1]^T
    float dM00_invy = R01 * R01;
    float dM01_invy = R01 * R11;
    float dM11_invy = R11 * R11;

    dinv_sx = (dM00*dM00_invx + 2*dM01*dM01_invx + dM11*dM11_invx) * (2*inv_sx);
    dinv_sy = (dM00*dM00_invy + 2*dM01*dM01_invy + dM11*dM11_invy) * (2*inv_sy);

    // ∂M/∂θ
    float dR00 = -sin_t, dR01 = -cos_t;
    float dR10 = cos_t, dR11 = -sin_t;

    float dm00_dtheta = 2*(dR00*inv_var_x*R00 + dR01*inv_var_y*R01);
    float dm01_dtheta = dR00*inv_var_x*R10 + R00*inv_var_x*dR10 +
                        dR01*inv_var_y*R11 + R01*inv_var_y*dR11;
    float dm11_dtheta = 2*(dR10*inv_var_x*R10 + dR11*inv_var_y*R11);

    dtheta = dM00*dm00_dtheta + 2*dM01*dm01_dtheta + dM11*dm11_dtheta;
}

__device__ inline void topk_selector(
    float* values,
    float* colors,
    int* topK_indices,
    int count,
    int K,
    int method  // 0=naive, 1=heap, 2=bitonic, 3=quickselect
) {
    // initialize local indices
    int indices[MAX_G_PER_TILE];
    for (int i = 0; i < count; i++) indices[i] = i;

    switch(method) {
        case 0: naive_sort(values, colors, indices, count); break;
        case 1: heap_topk(values, colors, indices, count, K); break;
        case 2: bitonic_topk(values, colors, indices, count, K); break;
        case 3: quickselect_topk(values, colors, indices, count, K); break;
        default: naive_sort(values, colors, indices, count); break;
    }

    int useK = min(K, count);
    for (int i = 0; i < useK; i++)
        topK_indices[i] = indices[i]; // return correct local indices
}

// CUDA kernel for tile-Gaussian correspondence
__global__ void findTileGaussianCorrespondence(
    const float* gaussian_means,    // [N_g, 2]
    const float* gaussian_rotations, // [N_g] (radians)
    const float* gaussian_log_scales, // [N_g, 2] (logs of scales)
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

        float log_sx = gaussian_log_scales[g * 2 + 0];
        float log_sy = gaussian_log_scales[g * 2 + 1];
        float sx = expf(log_sx);
        float sy = expf(log_sy);

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
    const float* gaussian_log_scales, // [N_g, 2] (logs of scales)
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

    // local buffers (assuming count ≤ max_G_per_tile)
    //extern __shared__ float shared[];
    //float* g_values = shared;
    //float* g_colors = g_values + max_G_per_tile;

    // CHANGED: per-thread local buffers (safe, no races).
    // WARNING: this uses register/local memory. If max_G_per_tile is large, this may spill to local memory.
    // Consider reducing max_G_per_tile or implementing chunked evaluation.
    float g_values[MAX_G_PER_TILE];          // MAX_G_PER_TILE compile-time limit
    float g_colors[MAX_G_PER_TILE * 3];

    // Evaluate Gaussians in this tile for this pixel (DEBUG GUARDS)
    for (int i = 0; i < count; ++i) {
        int g = g_indices[i];

        // Defensive check: g must be in [0, N_g) 
        if (g < 0 || g >= N_g) {
            printf("renderTile: BAD_G_INDEX g=%d at tile_idx=%d pixel=(%d,%d) count=%d N_g=%d n_tiles_x=%d\\n",
                   g, tile_idx, x, y, count, N_g, n_tiles_x);
            // set safe zero values
            g_values[i] = 0.0f;
            g_colors[i * 3 + 0] = 0.0f;
            g_colors[i * 3 + 1] = 0.0f;
            g_colors[i * 3 + 2] = 0.0f;
            continue;
        }

        // Normal (safe) code path
        float mu_x = gaussian_means[g * 2 + 0];
        float mu_y = gaussian_means[g * 2 + 1];
        float theta = gaussian_rotations[g];
        
        float log_sx = gaussian_log_scales[g * 2 + 0];
        float log_sy = gaussian_log_scales[g * 2 + 1];
        float sx = expf(log_sx);
        float sy = expf(log_sy);
        float inv_sx = 1.0f / sx;
        float inv_sy = 1.0f / sy;

        float val = gaussian_eval((float)x, (float)y, mu_x, mu_y, theta, inv_sx, inv_sy);
        g_values[i] = val;
        g_colors[i * 3 + 0] = gaussian_colors[g * 3 + 0];
        g_colors[i * 3 + 1] = gaussian_colors[g * 3 + 1];
        g_colors[i * 3 + 2] = gaussian_colors[g * 3 + 2];
    }

    // Option A: naive O(n^2)
    // naive_sort(g_values, g_colors, count);

    // Option B: heap-based Top-K
    //heap_topk(g_values, g_colors, count, K);

    // Option C: Bitonic TopK
    // bitonic_topk(g_values, g_colors, count, K);

    // Option D: QuickSelect
    //quickselect_topk(g_values, g_colors, count, K);

    int topK_indices[MAX_G_PER_TILE]; // temporary local storage
    topk_selector(g_values, g_colors, topK_indices, count, K, 2); // 0=naive, 1=heap, 2=bitonic, 3=quickselect

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
        pixel_topk_ptr[i] = local_idx; // store local index, NOT global
    }


        
    // Aggregate colors
    //int useK = min(K,count);
    float sum_weights = 0.0f;
    float sum_colors[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < useK; i++) {
        sum_weights += g_values[i];
        sum_colors[0] += g_values[i] * g_colors[i * 3 + 0];
        sum_colors[1] += g_values[i] * g_colors[i * 3 + 1];
        sum_colors[2] += g_values[i] * g_colors[i * 3 + 2];
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
    const float* gaussian_log_scales, // [N_g, 2] (logs of scales)
    const float* gaussian_colors,   // [N_g, 3]
    const int* tile_gaussian_indices,// [N_t, max_G_per_tile]
    const int* tile_gaussian_counts, // [N_t]
    float* grad_means,               // Gradient of means [N_g, 2]
    float* grad_rotations,           // Gradient of rotations [N_g]
    float* grad_log_scales,          // Gradient of inverse scales [N_g, 2]
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

    // temporary store per-tile values (stack allocation with compile-time bound)
    struct Gtmp {float val; int idx; float col[3]; float mu_x,mu_y,theta,inv_sx,inv_sy;};
    Gtmp gs[MAX_G_PER_TILE]; // assume max_G_per_tile <=MAX_G_PER_TILE

    for (int i = 0; i < count; i++) {
        int g=g_indices[i];
        
        if (g < 0 || g >= N_g) {
            printf("renderTileBackward 1: BAD_G_INDEX g=%d tile_idx=%d pixel=(%d,%d) count=%d N_g=%d\\n",
                   g, tile_idx, x, y, count, N_g);
            // mark invalid so later sorting/aggregation ignores it
            gs[i].val = 0.0f;
            gs[i].idx = -1;
            gs[i].col[0] = gs[i].col[1] = gs[i].col[2] = 0.0f;
            gs[i].mu_x = gs[i].mu_y = gs[i].theta = gs[i].inv_sx = gs[i].inv_sy = 0.0f;
            continue;
        }

        float mu_x=gaussian_means[g*2+0];
        float mu_y=gaussian_means[g*2+1];
        float theta=gaussian_rotations[g];
        
        float log_sx = gaussian_log_scales[g * 2 + 0];
        float log_sy = gaussian_log_scales[g * 2 + 1];
        float sx = expf(log_sx);
        float sy = expf(log_sy);
        float inv_sx = 1.0f / sx;
        float inv_sy = 1.0f / sy;

        float val=gaussian_eval(x,y,mu_x,mu_y,theta,inv_sx,inv_sy);

        // Store parameters
        gs[i].val=val; 
        gs[i].idx=g;

        gs[i].col[0]=gaussian_colors[g*3+0];
        gs[i].col[1]=gaussian_colors[g*3+1];
        gs[i].col[2]=gaussian_colors[g*3+2];

        gs[i].mu_x=mu_x; 
        gs[i].mu_y=mu_y; 

        gs[i].theta=theta; 

        gs[i].inv_sx=inv_sx; 
        gs[i].inv_sy=inv_sy;
    }

    // Sort descending by val (naive)
    //for(int i=0;i<count;i++)for(int j=i+1;j<count;j++)if(gs[i].val<gs[j].val){auto t=gs[i];gs[i]=gs[j];gs[j]=t;}
    // Option A: naive O(n^2)
    // naive_sort(g_values, g_colors, count);

    // Option B: heap-based Top-K
    //heap_topk(g_values, g_colors, count, K);

    // Option C: Bitonic TopK
    // bitonic_topk(g_values, g_colors, count, K);

    // Option D: QuickSelect
    //quickselect_topk(g_values, g_colors, count, K);



    // Aggregate colors and compute gradients

    int useK = min(K,count);
    const int* topK_indices = &pixel_topk_indices[(y*W + x)*K];
    // int useK = min(K, tile_gaussian_counts[tile_idx]); // count might be < K

    float sum_weights = 0.0f;
    float sum_colors[3] = {0.0f,0.0f,0.0f};

    for (int i = 0; i < useK; i++) {
        int g_local  = topK_indices[i];
        if (g_local  < 0 || g_local  >= N_g) {
            printf("renderTileBackward 2: BAD_G_INDEX g=%d at tile_idx=%d pixel=(%d,%d) count=%d N_g=%d n_tiles_x=%d\\n",
                   g_local , tile_idx, x, y, count, N_g, n_tiles_x);
            continue;
        }

        sum_weights += gs[g_local ].val;
        sum_colors[0] += gs[g_local ].val * gs[g_local ].col[0];
        sum_colors[1] += gs[g_local ].val * gs[g_local ].col[1];
        sum_colors[2] += gs[g_local ].val * gs[g_local ].col[2];
    }
    if (sum_weights < 1e-8f) return;


    // Reconstructed pixel color (matching forward)
    float I_c[3] = { sum_colors[0]/sum_weights, sum_colors[1]/sum_weights, sum_colors[2]/sum_weights };

    float go[3] = {grad_output[(y * W + x)*3 + 0], grad_output[(y * W + x)*3 + 1], grad_output[(y * W + x)*3 + 2]};

    // Aggregate colors
    for(int i=0;i<useK;i++){
        int g_local = topK_indices[i];
        if (g_local < 0 || g_local >= count) continue;

        int g = gs[g_local].idx;            // global Gaussian ID for writing gradients
        if (g < 0 || g >= N_g) continue;

        // Correct color gradient: dL/dc_k = (G_k / S) * go[c]
        float weight = gs[g_local].val / sum_weights;
        for(int c=0; c<3; c++){
            float add = weight * go[c];
            atomicAdd(&grad_colors[g*3 + c], add);
        }

        // Correct gradient wrt G_k:
        // dL/dG_k = sum_c go[c] * (c_k[c] - I_c[c]) / S
        float dL_dGk = 0.0f;
        for (int c=0; c<3; c++){
            dL_dGk += go[c] * (gs[g_local].col[c] - I_c[c]);
        }
        dL_dGk /= sum_weights;

        // Propagate grad_output through Gaussian eval
        float dmu_x, dmu_y, dtheta, dinv_sx, dinv_sy;
        gaussian_gradients((float)x, (float)y, gs[g_local].mu_x, gs[g_local].mu_y, gs[g_local].theta, gs[g_local].inv_sx, gs[g_local].inv_sy, 1.0f,
                           dmu_x, dmu_y, dtheta, dinv_sx, dinv_sy);

        // Multiply gradients by g_val
        dmu_x *= gs[g_local].val;
        dmu_y *= gs[g_local].val;
        dtheta *= gs[g_local].val;
        dinv_sx *= gs[g_local].val;
        dinv_sy *= gs[g_local].val;

        // Apply chain rule for log-scales
        float dlog_sx = -gs[g_local].inv_sx * dinv_sx;
        float dlog_sy = -gs[g_local].inv_sy * dinv_sy;

        // Apply gradient
        atomicAdd(&grad_means[g*2+0], dmu_x * dL_dGk);
        atomicAdd(&grad_means[g*2+1], dmu_y * dL_dGk);
        atomicAdd(&grad_rotations[g], dtheta * dL_dGk);
        atomicAdd(&grad_log_scales[g*2+0], dlog_sx  * dL_dGk);
        atomicAdd(&grad_log_scales[g*2+1], dlog_sy * dL_dGk);
    }
}


// ============================================================
// Wrappers
// ============================================================


// Wrapper functions for PyTorch
void find_tile_gaussian_correspondence_wrapper(
    torch::Tensor gaussian_means,
    torch::Tensor gaussian_rotations,
    torch::Tensor gaussian_log_scales,
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
    const float* gaussian_log_scales_ptr = gaussian_log_scales.data_ptr<float>();
    int* tile_gaussian_indices_ptr = tile_gaussian_indices.data_ptr<int>();
    int* tile_gaussian_counts_ptr = tile_gaussian_counts.data_ptr<int>();

    //dim3 block(16, 16);
    //dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
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
        gaussian_log_scales_ptr,
        N_g,
        H_t,
        W_t,
        H,
        W,
        tile_gaussian_indices_ptr,
        tile_gaussian_counts_ptr,
        max_G_per_tile
    );
    cudaDeviceSynchronize();
}

void render_tile_wrapper(
    torch::Tensor gaussian_means,
    torch::Tensor gaussian_rotations,
    torch::Tensor gaussian_log_scales,
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
    const float* gaussian_log_scales_ptr = gaussian_log_scales.data_ptr<float>();
    const float* gaussian_colors_ptr = gaussian_colors.data_ptr<float>();
    const int* tile_gaussian_indices_ptr = tile_gaussian_indices.data_ptr<int>();
    const int* tile_gaussian_counts_ptr = tile_gaussian_counts.data_ptr<int>();
    float* output_image_ptr = output_image.data_ptr<float>();
    int* pixel_topk_indices_ptr = pixel_topk_indices.data_ptr<int>();

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    
    //size_t shared_mem_size = max_G_per_tile * (1 + 3) * sizeof(float); // g_values + g_colors

    renderTile<<<grid, block>>>(
        gaussian_means_ptr,
        gaussian_rotations_ptr,
        gaussian_log_scales_ptr,
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
    cudaDeviceSynchronize();
}


// Wrapper for backward pass
void render_tile_backward_wrapper(
    torch::Tensor grad_output,
    torch::Tensor gaussian_means,
    torch::Tensor gaussian_rotations,
    torch::Tensor gaussian_log_scales,
    torch::Tensor gaussian_colors,
    torch::Tensor tile_gaussian_indices,
    torch::Tensor tile_gaussian_counts,
    torch::Tensor grad_means,
    torch::Tensor grad_rotations,
    torch::Tensor grad_log_scales,
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
    const float* gaussian_log_scales_ptr = gaussian_log_scales.data_ptr<float>();
    const float* gaussian_colors_ptr = gaussian_colors.data_ptr<float>();
    const int* tile_gaussian_indices_ptr = tile_gaussian_indices.data_ptr<int>();
    const int* tile_gaussian_counts_ptr = tile_gaussian_counts.data_ptr<int>();
    float* grad_means_ptr = grad_means.data_ptr<float>();
    float* grad_rotations_ptr = grad_rotations.data_ptr<float>();
    float* grad_log_scales_ptr = grad_log_scales.data_ptr<float>();
    float* grad_colors_ptr = grad_colors.data_ptr<float>();
    const int* pixel_topk_indices_ptr = pixel_topk_indices.data_ptr<int>();

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    renderTileBackward<<<grid, block>>>(
        grad_output_ptr,
        gaussian_means_ptr,
        gaussian_rotations_ptr,
        gaussian_log_scales_ptr,
        gaussian_colors_ptr,
        tile_gaussian_indices_ptr,
        tile_gaussian_counts_ptr,
        grad_means_ptr,
        grad_rotations_ptr,
        grad_log_scales_ptr,
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
    cudaDeviceSynchronize();
}


// Update PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_tile_gaussian_correspondence", &find_tile_gaussian_correspondence_wrapper, "Find tile-Gaussian correspondence");
    m.def("render_tile", &render_tile_wrapper, "Render tile");
    m.def("render_tile_backward", &render_tile_backward_wrapper, "Render tile backward");
}

