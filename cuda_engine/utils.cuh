// utils.cuh
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>
#include "sorting.cuh"

#define MAX_G_PER_TILE 128


// Utility: Gaussian evaluation
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


// Gaussian gradients
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