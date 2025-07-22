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
    const __global INPUT0_TYPE* restrict values,         // [N] - values at specified indices
    const __global INPUT1_TYPE* restrict dense_shape,    // [2] - shape of the dense tensor (rows, cols)
    const __global INPUT2_TYPE* restrict indices,        // [N, 2] - indices (row, col) coordinates
    const __global INPUT3_TYPE* restrict default_value,  // scalar - default value
    __global OUTPUT_TYPE* restrict output_indices,       // [M, 2] - output indices 
    __global OUTPUT1_TYPE* restrict output_values,       // [M] - output values
    __global OUTPUT2_TYPE* restrict empty_row_indicator  // [dense_shape[0]] - indicator if row was empty
) {
    const uint indices_count = INDICES_COUNT;  // JIT constant for N
    const uint num_rows = dense_shape[0];
    const uint row_idx = get_global_id(0);
    printf("Thread %d: Processing row %d of %d, indices_count=%d, default_value=%f\n", 
           get_global_id(0), row_idx, num_rows, indices_count, (float)*default_value);
    
    // Print input indices, values, dense_shape, and default_value
    printf("dense_shape: [%d, %d]\n", dense_shape[0], dense_shape[1]);
    printf("default_value: %f\n", (float)*default_value);
    for (uint i = 0; i < indices_count; i++) {
        printf("indices[%d] = [%d, %d], values[%d] = %f\n",
               i, indices[i * 2], indices[i * 2 + 1], i, (float)values[i]);
    }
    if (row_idx == 0) {
        printf("Input data - dense_shape: [%d, %d]\n", dense_shape[0], dense_shape[1]);
        for (uint i = 0; i < indices_count; i++) {
            printf("Input indices[%d] = [%d, %d], values[%d] = %f\n", 
                   i, indices[i * 2], indices[i * 2 + 1], i, (float)values[i]);
        }
    }
    
    // Check if this row is empty
    bool is_empty = !FUNC_CALL(row_exists)(indices, indices_count, row_idx);
    empty_row_indicator[row_idx] = is_empty;
    printf("Thread %d: Row %d is %s\n", get_global_id(0), row_idx, is_empty ? "EMPTY" : "NOT EMPTY");
    
    // We need to determine the exact output positions based on the row-major order
    // requirement of the operator specification
    
    // Calculate base position for this row in output
    uint output_base_pos = 0;
    
    // 1. Count all indices in rows before this one
    for (uint i = 0; i < indices_count; i++) {
        if (indices[i * 2] < row_idx) {
            output_base_pos++;
        }
    }
    
    // 2. Add empty rows before this one
    for (uint i = 0; i < row_idx; i++) {
        if (!FUNC_CALL(row_exists)(indices, indices_count, i)) {
            output_base_pos++;
        }
    }
    
    printf("Thread %d: Row %d base output position: %d\n", 
           get_global_id(0), row_idx, output_base_pos);
    
    // Handle empty and non-empty rows
    if (is_empty) {
        // Insert default value at [row_idx, 0]
        output_indices[output_base_pos * 2] = row_idx;
        output_indices[output_base_pos * 2 + 1] = 0;
        output_values[output_base_pos] = *default_value;
        
        printf("Thread %d: Set output_indices[%d] = [%d, %d], output_values[%d] = %f\n", 
               get_global_id(0), output_base_pos, output_indices[output_base_pos * 2], 
               output_indices[output_base_pos * 2 + 1], output_base_pos, (float)output_values[output_base_pos]);
    } else {
        // Find all entries for this row and place them in order
        uint row_pos = 0;
        for (uint i = 0; i < indices_count; i++) {
            if (indices[i * 2] == row_idx) {
                uint output_pos = output_base_pos + row_pos;
                
                // Copy indices and values
                output_indices[output_pos * 2] = indices[i * 2];
                output_indices[output_pos * 2 + 1] = indices[i * 2 + 1];
                output_values[output_pos] = values[i];
                
                printf("Thread %d: Set output_indices[%d] = [%d, %d], output_values[%d] = %f\n", 
                       get_global_id(0), output_pos, output_indices[output_pos * 2], 
                       output_indices[output_pos * 2 + 1], output_pos, (float)output_values[output_pos]);
                
                row_pos++;
            }
        }
    }
}
