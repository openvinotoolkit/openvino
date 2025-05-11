// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(extract_image_patches_ref)(const __global INPUT0_TYPE* input,
                                  __global OUTPUT_TYPE* output)
{
    const uint batch = (uint)get_global_id(0);
    const uint out_depth = (uint)get_global_id(1);
    const uint out_row = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint out_col = (uint)get_global_id(2) % OUTPUT_SIZE_X;

    int row_padding = 0;
    int col_padding = 0;
#ifdef AUTO_PAD
    uint num_out_rows = OUTPUT_SIZE_Y * STRIDE_ROWS + (SIZE_ROWS * RATES_ROWS - STRIDE_ROWS);
#if RATES_ROWS > 1
    --num_out_rows;
#endif // RATES_ROWS > 1
    const int row_padding_size = max((int)(num_out_rows - INPUT0_SIZE_Y), 0);
    uint num_out_cols = OUTPUT_SIZE_X * STRIDE_COLS + (SIZE_COLS * RATES_COLS - STRIDE_COLS);
#if RATES_COLS > 1
    --num_out_cols;
#endif // RATES_COLS > 1
    const int col_padding_size = max((int)(num_out_cols - INPUT0_SIZE_X), 0);
    row_padding = row_padding_size / 2;
    col_padding = col_padding_size / 2;
#if AUTO_PAD == 2 // same_lower
    row_padding = (row_padding_size % 2) + row_padding;
    col_padding = (col_padding_size % 2) + col_padding;
#endif // AUTO_PAD == 2
#endif // AUTO_PAD

    const uint cur_row_ind = out_depth / (INPUT0_FEATURE_NUM * SIZE_COLS);
    const uint row = cur_row_ind +
               STRIDE_ROWS * out_row - row_padding +
               (RATES_ROWS - 1) * cur_row_ind;
    const uint cur_col_ind = (out_depth % (INPUT0_FEATURE_NUM * SIZE_COLS)) / INPUT0_FEATURE_NUM;
    const uint col = cur_col_ind +
               STRIDE_COLS * out_col - col_padding +
               (RATES_COLS - 1) * cur_col_ind;

    const uint depth = out_depth % INPUT0_FEATURE_NUM;
    const uint in_ind = INPUT0_GET_INDEX_SAFE(batch, depth, row, col);
    const uint out_ind = OUTPUT_GET_INDEX(batch, out_depth, out_row, out_col);
    OUTPUT_TYPE res = TO_OUTPUT_TYPE(input[in_ind]);
#ifdef AUTO_PAD
    if (row < 0 || col < 0 || row >= INPUT0_SIZE_Y || col >= INPUT0_SIZE_X)
        res = OUTPUT_VAL_ZERO;
#endif
    output[out_ind] = ACTIVATION(res, ACTIVATION_PARAMS);
}
