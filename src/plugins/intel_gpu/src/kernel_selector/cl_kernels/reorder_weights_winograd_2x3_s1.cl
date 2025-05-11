// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(reorder_weights_winograd_2x3_s1)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
#if OUTPUT_LAYOUT_WINOGRAD_2x3_S1_WEIGHTS
    const uint input_tile_width = 3;
    const uint input_tile_height = 1;
    const uint in_tile_x_idx = get_global_id(0);
    const uint in_tile_y_idx = get_global_id(1);
#else //OUTPUT_LAYOUT_WINOGRAD_2x3_S1_FUSED_WEIGHTS
    const uint input_tile_width = 1;
    const uint input_tile_height = 3;
    const uint in_tile_x_idx = get_global_id(1);
    const uint in_tile_y_idx = get_global_id(0);
#endif

    const uint output_tile_width = 4;
    const uint output_tile_height = 1;

    const uint tile_x_idx = get_global_id(0);
    const uint tile_y_idx = get_global_id(1);
    const uint feature_idx = (uint)get_global_id(2) % INPUT0_IFM_NUM;
    const uint batch_idx = (uint)get_global_id(2) / INPUT0_IFM_NUM;

    uint in_idx = batch_idx * INPUT0_OFM_PITCH
                 + feature_idx * INPUT0_IFM_PITCH
                 + in_tile_y_idx * input_tile_height * INPUT0_Y_PITCH
                 + in_tile_x_idx * input_tile_width * INPUT0_X_PITCH;

#if OUTPUT_LAYOUT_WINOGRAD_2x3_S1_WEIGHTS
    MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) tile;
    tile.x = input[in_idx]; in_idx += INPUT0_X_PITCH;
    tile.y = input[in_idx]; in_idx += INPUT0_X_PITCH;
    tile.z = input[in_idx];

    uint out_idx = batch_idx * OUTPUT_OFM_PITCH
                  + feature_idx * OUTPUT_IFM_PITCH
                  + tile_y_idx * output_tile_height * OUTPUT_Y_PITCH
                  + tile_x_idx * output_tile_width * OUTPUT_X_PITCH;

    output[out_idx] = TO_OUTPUT_TYPE(tile.x); out_idx += OUTPUT_X_PITCH;
    output[out_idx] = TO_OUTPUT_TYPE((tile.x + tile.y + tile.z) / 2.0f); out_idx += OUTPUT_X_PITCH;
    output[out_idx] = TO_OUTPUT_TYPE((tile.x - tile.y + tile.z) / 2.0f); out_idx += OUTPUT_X_PITCH;
    output[out_idx] = TO_OUTPUT_TYPE(tile.z);
#else //OUTPUT_LAYOUT_WINOGRAD_2x3_S1_FUSED_WEIGHTS
    MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) tile;
    tile.x = input[in_idx]; in_idx += INPUT0_Y_PITCH;
    tile.y = input[in_idx]; in_idx += INPUT0_Y_PITCH;
    tile.z = input[in_idx];

    const uint weightsOSplit = 8;
    const uint oDivSplit = OUTPUT_OFM_NUM / 8;
    uint out_idx = batch_idx % 8 + tile_y_idx * output_tile_height * weightsOSplit +
        tile_x_idx * output_tile_width * weightsOSplit * OUTPUT_SIZE_Y +
        batch_idx / 8 * weightsOSplit * OUTPUT_SIZE_X * OUTPUT_SIZE_Y +
        feature_idx * weightsOSplit * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * oDivSplit;

    output[out_idx] = TO_OUTPUT_TYPE(tile.x); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
    output[out_idx] = TO_OUTPUT_TYPE((tile.x + tile.y + tile.z) / 2.0f); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
    output[out_idx] = TO_OUTPUT_TYPE((tile.x - tile.y + tile.z) / 2.0f); out_idx += weightsOSplit * OUTPUT_SIZE_Y;
    output[out_idx] = TO_OUTPUT_TYPE(tile.z);
#endif
}
