// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// --------------------------------------------------------------------------------------------------------------------------------
// Convert the results using the inverse F(2,3) Winograd transform.
// --------------------------------------------------------------------------------------------------------------------------------

#include "include/batch_headers/fetch_data.cl"

KERNEL(reorder_from_winograd_2x3_s1)(global const UNIT_TYPE* input_winograd, global float* output)
{
    const int winograd_tile_width = 4;
    const int winograd_tile_height = 1;
    const int output_tile_width = 2;
    const int output_tile_height = 1;

    const int batch_idx = (uint)get_global_id(0) / INPUT0_FEATURE_NUM;
    const int feature_idx = (uint)get_global_id(0) % INPUT0_FEATURE_NUM;
    const int tile_idx_x = get_global_id(1);
    const int tile_idx_y = get_global_id(2);

    const int out_x_idx = (tile_idx_x * output_tile_width);

    //input is in bxyf -- no paddings allowed in winograd domain
    int input_idx = batch_idx * INPUT0_BATCH_PITCH +
                    feature_idx * INPUT0_FEATURE_PITCH +
                    tile_idx_y * winograd_tile_height * INPUT0_Y_PITCH +
                    tile_idx_x * winograd_tile_width * INPUT0_X_PITCH;

    //winograd tile is 4x1, during conversion to standard domain values should have already been multiplied so this tile is actually an 'm' tile from the original paper
    UNIT_TYPE winograd_tile[winograd_tile_width];
    winograd_tile[0] = input_winograd[input_idx]; input_idx += INPUT0_X_PITCH;
    winograd_tile[1] = input_winograd[input_idx]; input_idx += INPUT0_X_PITCH;
    winograd_tile[2] = input_winograd[input_idx]; input_idx += INPUT0_X_PITCH;
    winograd_tile[3] = input_winograd[input_idx];

    UNIT_TYPE out_tile[output_tile_width];

    //transform back
    out_tile[0] = ACTIVATION(winograd_tile[0] + winograd_tile[1] + winograd_tile[2], ACTIVATION_PARAMS);
    out_tile[1] = ACTIVATION(winograd_tile[1] - winograd_tile[2] - winograd_tile[3], ACTIVATION_PARAMS);

    int out_idx = (OUTPUT_PAD_BEFORE_BATCH_NUM + batch_idx) * OUTPUT_BATCH_PITCH +
                  (OUTPUT_PAD_BEFORE_FEATURE_NUM + feature_idx) * OUTPUT_FEATURE_PITCH +
                  (OUTPUT_PAD_BEFORE_SIZE_Y + (tile_idx_y * output_tile_height)) * OUTPUT_Y_PITCH +
                  (OUTPUT_PAD_BEFORE_SIZE_X + (tile_idx_x * output_tile_width)) * OUTPUT_X_PITCH;

    output[out_idx] = out_tile[0];
#if LEFTOVERS == 1
    if (out_x_idx + 1 < OUTPUT_SIZE_X)
#endif
    {
        out_idx += OUTPUT_X_PITCH;
        output[out_idx] = out_tile[1];
    }
};
