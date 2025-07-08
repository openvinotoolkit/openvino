// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(sparse_fill_empty_rows_ref)(
    const __global INPUT0_TYPE* restrict indices,        // [N, 2] - indices (row, col) coordinates
    const __global INPUT1_TYPE* restrict values,         // [N] - values at specified indices
    const __global INPUT2_TYPE* restrict dense_shape,    // [2] - shape of the dense tensor (rows, cols)
    const __global INPUT3_TYPE* restrict default_value,  // scalar - default value
    __global OUTPUT_TYPE* restrict output_indices,       // [M, 2] - output indices 
    __global OUTPUT1_TYPE* restrict output_values,       // [M] - output values
    __global OUTPUT2_TYPE* restrict empty_row_indicator  // [dense_shape[0]] - indicator if row was empty
) {
    printf("SparseFillEmptyRows kernel is not implemented for this device.\n");
}
