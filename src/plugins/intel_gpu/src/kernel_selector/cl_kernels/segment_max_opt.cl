// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// SegmentMax optimized kernel.
//
// Leverages the SegmentMax-16 specification requirement that segment_ids is
// sorted (non-decreasing) to use binary search for finding segment boundaries.
// This reduces the per-output-element complexity from O(num_rows) to
// O(log(num_rows) + segment_size).
//
// Work distribution:
//   get_global_id(0) = inner dimension index j  (coalesced memory access)
//   get_global_id(1) = segment index
//
// Inputs:
//   data         – [num_rows x inner_dim_size] flattened
//   segment_ids  – [num_rows] sorted non-decreasing
// Output:
//   output       – [num_segments x inner_dim_size]

// Binary search: find first index in [lo, hi) where segment_ids[index] >= target.
inline int __attribute__((always_inline)) lower_bound(
    const __global INPUT1_TYPE* restrict segment_ids,
    int lo, int hi, int target)
{
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if ((int)segment_ids[mid] < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Binary search: find first index in [lo, hi) where segment_ids[index] > target.
inline int __attribute__((always_inline)) upper_bound(
    const __global INPUT1_TYPE* restrict segment_ids,
    int lo, int hi, int target)
{
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if ((int)segment_ids[mid] <= target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

KERNEL(segment_max_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict data,
    const __global INPUT1_TYPE* restrict segment_ids,
    __global OUTPUT_TYPE* restrict output)
{
    const int j   = get_global_id(0);   // inner dimension index (coalesced access)
    const int seg = get_global_id(1);   // segment index

    const int num_segments   = OUTPUT_BATCH_NUM;
    const int inner_dim_size = OUTPUT_FEATURE_NUM * OUTPUT_SIZE_Z * OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const int num_rows       = INPUT0_BATCH_NUM;

    if (j >= inner_dim_size || seg >= num_segments)
        return;

    // Binary search for the half-open range [start, end) of rows belonging
    // to this segment.  O(log num_rows) instead of O(num_rows).
    const int start = lower_bound(segment_ids, 0, num_rows, seg);
    const int end   = upper_bound(segment_ids, start, num_rows, seg);

    const int out_idx = seg * inner_dim_size + j;

    if (start >= end) {
        // Empty segment — fill with the appropriate sentinel.
        output[out_idx] = EMPTY_SEGMENT_VALUE;
        return;
    }

    // Reduce over the segment rows.
    // First element is unconditionally assigned, avoiding a branch in the loop.
    OUTPUT_TYPE max_val = (OUTPUT_TYPE)data[start * inner_dim_size + j];

    for (int i = start + 1; i < end; ++i) {
        const INPUT0_TYPE v = data[i * inner_dim_size + j];
        if (v > max_val) {
            max_val = (OUTPUT_TYPE)v;
        }
    }

    output[out_idx] = max_val;
}
