// sorting.cuh
#pragma once
#include <cuda_runtime.h>

#define MAX_G_PER_TILE 128

// convert float to sortable unsigned int
__device__ inline unsigned int float_as_uint(float f) {
    unsigned int x = *(unsigned int*)&f;
    return (f >= 0) ? x : x ^ 0xFFFFFFFF;
}

// Radix-inspired top-K for small arrays using shared memory
__device__ inline void radix_topk_shared(
    const float* values,      // per-thread evaluated values
    int* local_indices,       // initial [0..count-1]
    int count,
    int K
) {
    // iterative max-selection
    for (int k = 0; k < K; ++k) {
        float max_val = -1e30f;
        int max_idx = -1;
        for (int i = k; i < count; ++i) { // start at k
            float v = values[local_indices[i]];
            if (v > max_val) {
                max_val = v;
                max_idx = i;
            }
        }
        if (max_idx >= 0) {
            // swap in local_indices only
            int tmp = local_indices[k];
            local_indices[k] = local_indices[max_idx];
            local_indices[max_idx] = tmp;
        }
    }
}

__device__ inline void topk_selector(
    float* values,
    int* topK_indices,
    int count,
    int K
) {
    // initialize local indices
    for (int i = 0; i < count; i++) topK_indices[i] = i;
    radix_topk_shared(values, topK_indices, count, K);
}
