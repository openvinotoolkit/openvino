// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

inline bool FUNC(row_exists)(const __global INPUT2_TYPE* restrict indices, uint indices_count, INPUT2_TYPE row_idx) {
    for (uint i = 0; i < indices_count; i++) {
        if (indices[i * 2] == row_idx) {
            return true;
        }
    }
    return false;
}

KERNEL(sparse_fill_empty_rows_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict values,         // [N] - values at specified indices
    const __global INPUT1_TYPE* restrict _unused,        // [2] - shape of the dense tensor (rows, cols)
    const __global INPUT2_TYPE* restrict indices,        // [N, 2] - indices (row, col) coordinates
    const __global INPUT3_TYPE* restrict default_value,  // scalar - default value
    __global OUTPUT_TYPE* restrict output_indices,       // [M, 2] - output indices 
    __global OUTPUT1_TYPE* restrict output_values,       // [M] - output values
    __global OUTPUT2_TYPE* restrict empty_row_indicator  // [dense_shape[0]] - indicator if row was empty
) {
    const uint indices_count = INPUT2_BATCH_NUM;
    const uint row_idx = get_global_id(0);
    uint output_base_pos = 0;

    // Count all indices in rows before this one
    for (uint i = 0; i < indices_count; i++) {
        if (indices[i * 2] < row_idx) {
            output_base_pos++;
        }
    }
    // Add empty rows before this one
    for (uint i = 0; i < row_idx; i++) {
        if (!FUNC_CALL(row_exists)(indices, indices_count, i)) {
            output_base_pos++;
        }
    }

    const bool is_empty = !FUNC_CALL(row_exists)(indices, indices_count, row_idx);
    empty_row_indicator[row_idx] = is_empty;
    if (is_empty) {
        // Insert default value at [row_idx, 0]
        output_indices[output_base_pos * 2] = row_idx;
        output_indices[output_base_pos * 2 + 1] = 0;
        output_values[output_base_pos] = *default_value;
    } else {
        // Find all existing entries for this row and place them in order
        uint row_pos = 0;
        for (uint i = 0; i < indices_count; i++) {
            if (indices[i * 2] == row_idx) {
                uint output_pos = output_base_pos + row_pos;
                output_indices[output_pos * 2] = indices[i * 2];
                output_indices[output_pos * 2 + 1] = indices[i * 2 + 1];
                output_values[output_pos] = values[i];
                row_pos++;
            }
        }
    }
}
