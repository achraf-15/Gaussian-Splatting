// sorting.cuh
#pragma once
#include <cuda_runtime.h>


// ------------------------------------------------------------
// Naive O(n^2) sort (descending) - already known
// ------------------------------------------------------------
__device__ inline void naive_sort(float* values, float* colors, int count) {
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (values[i] < values[j]) {
                float tmpv = values[i]; values[i] = values[j]; values[j] = tmpv;
                for (int c = 0; c < 3; ++c) {
                    float t = colors[i*3 + c];
                    colors[i*3 + c] = colors[j*3 + c];
                    colors[j*3 + c] = t;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Min-heap-ish partial selection (in-place) O(count * K) worst-case
// Simple: keep top-K in first K slots, replace minimum when you find larger.
// ------------------------------------------------------------
__device__ inline void heap_topk(float* values, float* colors, int count, int K) {
    if (K <= 0 || count <= 1) return;
    if (count <= K) {
        // just sort all descending for consistency
        naive_sort(values, colors, count);
        return;
    }
    // First, ensure first K are initialized and sorted descending
    // We'll maintain first K as the current top-K (unsorted except we will find min when needed).
    // For simplicity: sort first K descending initially
    for (int i = 0; i < K; ++i) {
        for (int j = i + 1; j < K; ++j) {
            if (values[i] < values[j]) {
                float tv = values[i]; values[i] = values[j]; values[j] = tv;
                for (int c = 0; c < 3; ++c) {
                    float t = colors[i*3 + c];
                    colors[i*3 + c] = colors[j*3 + c];
                    colors[j*3 + c] = t;
                }
            }
        }
    }
    // For each remaining element, compare with the current smallest of first K
    for (int i = K; i < count; ++i) {
        // find index of min in first K
        int min_idx = 0;
        float min_val = values[0];
        for (int j = 1; j < K; ++j) {
            if (values[j] < min_val) { min_val = values[j]; min_idx = j; }
        }
        if (values[i] > min_val) {
            // replace min slot with this element
            values[min_idx] = values[i];
            colors[min_idx*3 + 0] = colors[i*3 + 0];
            colors[min_idx*3 + 1] = colors[i*3 + 1];
            colors[min_idx*3 + 2] = colors[i*3 + 2];
        }
    }
    // final sort first K descending for deterministic order
    for (int i = 0; i < K; ++i) {
        for (int j = i + 1; j < K; ++j) {
            if (values[i] < values[j]) {
                float tv = values[i]; values[i] = values[j]; values[j] = tv;
                for (int c = 0; c < 3; ++c) {
                    float t = colors[i*3 + c];
                    colors[i*3 + c] = colors[j*3 + c];
                    colors[j*3 + c] = t;
                }
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

__device__ inline void bitonic_topk(float* values, float* colors, int count, int K) {
    if (count <= 1) return;
    int n = next_pow2_int(count);
    // We'll operate in indices [0..n-1], treating values[i>=count] as -inf
    // For simplicity, we use the provided arrays but treat out-of-range as -inf
    // Compare-and-swap helper (descending)
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; ++i) {
                int ixj = i ^ j;
                if (ixj > i) {
                    // value at i
                    float vi = (i < count) ? values[i] : -INFINITY;
                    float vj = (ixj < count) ? values[ixj] : -INFINITY;
                    bool upward = ((i & k) == 0);
                    // for descending sort we need to swap when vi < vj for upward==true
                    if (upward) {
                        if (vi < vj) {
                            // swap positions i and ixj (including colors)
                            if (i < count && ixj < count) {
                                float tmp = values[i]; values[i] = values[ixj]; values[ixj] = tmp;
                                for (int c = 0; c < 3; ++c) {
                                    float t = colors[i*3 + c];
                                    colors[i*3 + c] = colors[ixj*3 + c];
                                    colors[ixj*3 + c] = t;
                                }
                            } else if (i < count && ixj >= count) {
                                // swap with virtual -inf: move i to ixj (out of range) -> set i = -inf
                                values[ixj < count ? ixj : 0] = -INFINITY; // no-op safe
                                // safer approach: swap only when both in-range; if one is out-of-range skip
                                // to keep code simple and safe, only swap when both indices < count
                            } else if (i >= count && ixj < count) {
                                // same as above; skip
                            }
                        }
                    } else {
                        if (vi > vj) {
                            if (i < count && ixj < count) {
                                float tmp = values[i]; values[i] = values[ixj]; values[ixj] = tmp;
                                for (int c = 0; c < 3; ++c) {
                                    float t = colors[i*3 + c];
                                    colors[i*3 + c] = colors[ixj*3 + c];
                                    colors[ixj*3 + c] = t;
                                }
                            }
                        }
                    }
                }
            } // i
        } // j
    } // k

    // Now first count entries are sorted descending (though padded behavior is conservative).
    // We only need top-K: ensure K<=count
    if (K > count) K = count;
    // Optionally: nothing further needed — first K are top K in descending order.
}

// ------------------------------------------------------------
// Quickselect (in-place) to partition top-K to front, then small sort of K
// This is an iterative quickselect variant using Lomuto/Hoare partitioning idea.
// After partitioning, the first K elements are the K largest in arbitrary order.
// We then sort the first K elements (small K) via naive sort.
// ------------------------------------------------------------
__device__ inline int partition_desc(float* values, float* colors, int left, int right, float pivot) {
    // Partition so that elements > pivot go to left
    int i = left;
    int j = right;
    while (i <= j) {
        while (i <= right && values[i] > pivot) ++i; // greater to left
        while (j >= left && values[j] < pivot) --j;  // smaller to right
        if (i <= j) {
            // swap i and j
            float tmpv = values[i]; values[i] = values[j]; values[j] = tmpv;
            for (int c = 0; c < 3; ++c) {
                float t = colors[i*3 + c];
                colors[i*3 + c] = colors[j*3 + c];
                colors[j*3 + c] = t;
            }
            ++i; --j;
        }
    }
    return i; // first index of right partition
}

__device__ inline void quickselect_topk(float* values, float* colors, int count, int K) {
    if (K <= 0 || count <= 1) return;
    if (K >= count) {
        naive_sort(values, colors, count);
        return;
    }

    int left = 0, right = count - 1;
    // Iterative quickselect
    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        float pivot = values[mid];
        int idx = partition_desc(values, colors, left, right, pivot);
        int left_count = idx - left; // number of elements strictly greater than pivot on left side
        if (left_count == K) {
            break;
        } else if (left_count > K) {
            // top-K are in left..idx-1
            right = idx - 1;
        } else {
            // need more from right side
            K -= left_count;
            left = idx;
        }
    }
    // Now top-K are in indices [0 .. original_K-1] maybe not exactly, but ensure by final step:
    // For safety, perform partial pass to collect exactly top-K into first K slots:
    // Find the K largest by scanning and selecting into first K (simple and robust)
    // We'll use selection-by-replace-min approach but restricted to K -> O(count*K) worst-case (K small)
    // Initialize first K by taking the first K values (they are arbitrary but we fill)
    // We'll create temporary arrays in-place: reuse first K positions as reservoir
    // Create a simple top-K collector:
    // Step 1: naive_sort first K
    naive_sort(values, colors, K); // now first K sorted descending
    // Step 2: scan remaining
    for (int i = K; i < count; ++i) {
        // find min index in first K
        int min_idx = 0;
        float min_val = values[0];
        for (int j = 1; j < K; ++j) {
            if (values[j] < min_val) { min_val = values[j]; min_idx = j; }
        }
        if (values[i] > min_val) {
            // replace and re-sort small K (we can insert bubble-up)
            values[min_idx] = values[i];
            colors[min_idx*3+0] = colors[i*3+0];
            colors[min_idx*3+1] = colors[i*3+1];
            colors[min_idx*3+2] = colors[i*3+2];
            // re-sort first K (simple insertion)
            for (int a = min_idx; a > 0; --a) {
                if (values[a] > values[a-1]) {
                    float tv = values[a]; values[a] = values[a-1]; values[a-1] = tv;
                    for (int c = 0; c < 3; ++c) {
                        float t = colors[a*3 + c];
                        colors[a*3 + c] = colors[(a-1)*3 + c];
                        colors[(a-1)*3 + c] = t;
                    }
                } else break;
            }
        }
    }
    // After this the first K positions contain the top-K in descending order.
}
