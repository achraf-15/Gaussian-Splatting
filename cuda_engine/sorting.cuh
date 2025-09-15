// sorting.cuh
#pragma once
#include <cuda_runtime.h>


// ------------------------------------------------------------
// Naive O(n^2) sort (descending) - already known
// ------------------------------------------------------------
__device__ inline void naive_sort(float* values, float* colors, int* indices, int count) {
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (values[i] < values[j]) {
                // swap values
                float tmpv = values[i]; values[i] = values[j]; values[j] = tmpv;
                // swap colors
                for (int c = 0; c < 3; ++c) {
                    float t = colors[i*3 + c]; colors[i*3 + c] = colors[j*3 + c]; colors[j*3 + c] = t;
                }
                // swap indices
                int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
            }
        }
    }
}


// ------------------------------------------------------------
// Min-heap-ish partial selection (in-place) O(count * K) worst-case
// Simple: keep top-K in first K slots, replace minimum when you find larger.
// ------------------------------------------------------------
__device__ inline void heap_topk(float* values, float* colors, int* indices, int count, int K) {
    if (K <= 0 || count <= 1) return;
    if (count <= K) { naive_sort(values, colors, indices, count); return; }

    // initialize indices if not already done
    for (int i = 0; i < count; i++) indices[i] = i;

    // first K as top-K (descending)
    naive_sort(values, colors, indices, K);

    for (int i = K; i < count; ++i) {
        // find min in first K
        int min_idx = 0;
        float min_val = values[0];
        for (int j = 1; j < K; ++j) {
            if (values[j] < min_val) { min_val = values[j]; min_idx = j; }
        }
        if (values[i] > min_val) {
            // replace
            values[min_idx] = values[i];
            for (int c = 0; c < 3; ++c)
                colors[min_idx*3 + c] = colors[i*3 + c];
            indices[min_idx] = indices[i];
            // re-sort first K
            for (int a = min_idx; a > 0; --a) {
                if (values[a] > values[a-1]) {
                    float tv = values[a]; values[a] = values[a-1]; values[a-1] = tv;
                    for (int c = 0; c < 3; ++c) {
                        float t = colors[a*3 + c]; colors[a*3 + c] = colors[(a-1)*3 + c]; colors[(a-1)*3 + c] = t;
                    }
                    int ti = indices[a]; indices[a] = indices[a-1]; indices[a-1] = ti;
                } else break;
            }
        }
    }
}

// ------------------------------------------------------------
// Bitonic sort (in-place) — sorts the whole array descending.
// Works by padding to next power-of-two; extra slots are set to -inf.
// Good for small arrays; predictable performance.
// ------------------------------------------------------------
__device__ inline int next_pow2_int(int v) {
    int n = 1;
    while (n < v) n <<= 1;
    return n;
}

__device__ inline void bitonic_topk(float* values, float* colors, int* indices, int count, int K) {
    if (count <= 1) return;
    int n = next_pow2_int(count);
    // Compare-and-swap helper (descending)
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; ++i) {
                int ixj = i ^ j;
                if (ixj > i && i < count && ixj < count) {
                    bool upward = ((i & k) == 0);
                    float vi = values[i];
                    float vj = values[ixj];
                    bool swap_needed = (upward && vi < vj) || (!upward && vi > vj);
                    if (swap_needed) {
                        // swap values
                        float tmp = values[i]; values[i] = values[ixj]; values[ixj] = tmp;
                        // swap colors
                        for (int c = 0; c < 3; ++c) {
                            float t = colors[i*3 + c]; colors[i*3 + c] = colors[ixj*3 + c]; colors[ixj*3 + c] = t;
                        }
                        // swap indices
                        int ti = indices[i]; indices[i] = indices[ixj]; indices[ixj] = ti;
                    }
                }
            }
        }
    }

    // Only top-K are needed; K <= count
    if (K > count) K = count;
    // First K entries now contain top-K in descending order
}

// ------------------------------------------------------------
// Quickselect (in-place) to partition top-K to front, then small sort of K
// This is an iterative quickselect variant using Lomuto/Hoare partitioning idea.
// After partitioning, the first K elements are the K largest in arbitrary order.
// We then sort the first K elements (small K) via naive sort.
// ------------------------------------------------------------
__device__ inline int partition_desc(float* values, float* colors, int* indices, int left, int right, float pivot) {
    int i = left, j = right;
    while (i <= j) {
        while (i <= right && values[i] > pivot) ++i;
        while (j >= left && values[j] < pivot) --j;
        if (i <= j) {
            // swap values
            float tmpv = values[i]; values[i] = values[j]; values[j] = tmpv;
            // swap colors
            for (int c = 0; c < 3; ++c) {
                float t = colors[i*3 + c]; colors[i*3 + c] = colors[j*3 + c]; colors[j*3 + c] = t;
            }
            // swap indices
            int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;

            ++i; --j;
        }
    }
    return i;
}

__device__ inline void quickselect_topk(float* values, float* colors, int* indices, int count, int K) {
    if (K <= 0 || count <= 1) return;
    if (K >= count) { naive_sort(values, colors, indices, count); return; }

    int left = 0, right = count - 1;
    int original_K = K;

    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        float pivot = values[mid];
        int idx = partition_desc(values, colors, indices, left, right, pivot);
        int left_count = idx - left;

        if (left_count == K) break;
        else if (left_count > K) right = idx - 1;
        else {
            K -= left_count;
            left = idx;
        }
    }

    // Ensure first original_K positions contain exact top-K
    // Use simple selection: naive_sort first original_K and insert remaining if needed
    naive_sort(values, colors, indices, original_K);
    for (int i = original_K; i < count; ++i) {
        // find min in first original_K
        int min_idx = 0; float min_val = values[0];
        for (int j = 1; j < original_K; ++j) if (values[j] < min_val) { min_val = values[j]; min_idx = j; }
        if (values[i] > min_val) {
            values[min_idx] = values[i];
            for (int c = 0; c < 3; ++c) colors[min_idx*3 + c] = colors[i*3 + c];
            indices[min_idx] = indices[i];

            // bubble-up to maintain descending order
            for (int a = min_idx; a > 0; --a) {
                if (values[a] > values[a-1]) {
                    float tv = values[a]; values[a] = values[a-1]; values[a-1] = tv;
                    for (int c = 0; c < 3; ++c) {
                        float t = colors[a*3 + c]; colors[a*3 + c] = colors[(a-1)*3 + c]; colors[(a-1)*3 + c] = t;
                    }
                    int ti = indices[a]; indices[a] = indices[a-1]; indices[a-1] = ti;
                } else break;
            }
        }
    }
}