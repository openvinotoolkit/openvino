// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// SegmentMax reference kernel.
//
// For each output element (segment_id, j) where j is an index in the inner
// dimension, scans all data rows that belong to the segment and computes the
// maximum value.
//
// Inputs:
//   data         – [num_rows x inner_dim_size] flattened
//   segment_ids  – [num_rows]
// Output:
//   output       – [num_segments x inner_dim_size]

KERNEL(segment_max_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict data,
    const __global INPUT1_TYPE* restrict segment_ids,
    __global OUTPUT_TYPE* restrict output)
{
    const int gid = get_global_id(0);

    // Total number of output elements = num_segments * inner_dim_size
    const int num_segments   = OUTPUT_BATCH_NUM;
    const int inner_dim_size = OUTPUT_FEATURE_NUM * OUTPUT_SIZE_Z * OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const int total_output   = num_segments * inner_dim_size;

    if (gid >= total_output)
        return;

    const int seg = gid / inner_dim_size;
    const int j   = gid % inner_dim_size;

    // num_rows is the first dimension of the data tensor
    const int num_rows = INPUT0_BATCH_NUM;

    OUTPUT_TYPE max_val = EMPTY_SEGMENT_VALUE;
    int found = 0;

    for (int i = 0; i < num_rows; ++i) {
        if ((INPUT1_TYPE)segment_ids[i] == (INPUT1_TYPE)seg) {
            INPUT0_TYPE v = data[i * inner_dim_size + j];
            if (!found || v > max_val) {
                max_val = (OUTPUT_TYPE)v;
            }
            found = 1;
        }
    }

    if (!found) {
        max_val = EMPTY_SEGMENT_VALUE;
    }

    output[gid] = max_val;
}
