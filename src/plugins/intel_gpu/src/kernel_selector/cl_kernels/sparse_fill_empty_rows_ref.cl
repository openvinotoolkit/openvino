// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// indices = [[row1, col1], [row2, col2], ...]
// dense_shape = [num_rows, num_cols]
// values = [val1, val2, ...]

// before call: fill empty_row_indicator with 0s
KERNEL(sparse_fill_empty_rows_find_empty_rows)(
    const __global INPUT0_TYPE* restrict indices,
    __global OUTPUT_TYPE* restrict empty_row_indicator)
{
    const uint d1 = get_global_id(0);
    const uint indices_row = indices[2 * d1];
    empty_row_indicator[indices_row] = 1;
}

// pass empty_row_indicator
KERNEL(sparse_fill_empty_rows_prefix_sum)(
    const __global int* to_insert,
    __global int* scan) {
    int lid = get_local_id(0);
    int prefix = work_group_scan_exclusive_add(to_insert[lid]);
    scan[lid] = prefix;
}

// based on empty_row_indicator extend the values array on CPU side
KERNEL(sparse_fill_empty_rows_ref)(
    const __global INPUT0_TYPE* restrict indices,
    const __global INPUT1_TYPE* restrict empty_row_indicator,
    const __global uint* restrict prefix_sum,
    const uint num_input_indices,
    __global OUTPUT_TYPE0* restrict output_indices)
{
    const uint gid = get_global_id(0);
    if (gid < num_input_indices) {
        const uint row = indices[2 * gid];
        const uint out_pos = gid + prefix_sum[row];
        output_indices[2 * out_pos] = row;
        output_indices[2 * out_pos + 1] = indices[2 * gid + 1];
    } else {
        // Handle empty row entry
        uint empty_idx = gid - num_input_indices;
        uint row = 0;
        while (!(empty_row_indicator[row] == 0 && prefix_sum[row] == empty_idx)) {
            ++row;
        }
        uint out_pos = prefix_sum[row] + row;
        output_indices[2 * out_pos] = row;
        output_indices[2 * out_pos + 1] = 0;
    }
}
