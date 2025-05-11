// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(reorder_weights_winograd_6x3_s1)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint input_tile_width = 1;
    const uint input_tile_height = 3;
    const uint in_tile_x_idx = get_global_id(1);
    const uint in_tile_y_idx = get_global_id(0);

    const uint output_tile_width = 8;
    const uint output_tile_height = 1;

    const uint tile_x_idx = get_global_id(0);
    const uint tile_y_idx = get_global_id(1);
    const uint feature_idx = (uint)get_global_id(2) % INPUT0_IFM_NUM;
    const uint batch_idx = (uint)get_global_id(2) / INPUT0_IFM_NUM;

    uint in_idx = batch_idx * INPUT0_OFM_PITCH
                 + feature_idx * INPUT0_IFM_PITCH
                 + in_tile_y_idx * input_tile_height * INPUT0_Y_PITCH
                 + in_tile_x_idx * input_tile_width * INPUT0_X_PITCH;

    MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) tile;
    tile.x = input[in_idx]; in_idx += INPUT0_Y_PITCH;
    tile.y = input[in_idx]; in_idx += INPUT0_Y_PITCH;
    tile.z = input[in_idx];

    const uint weightsOSplit = 16;
    const uint oDivSplit = OUTPUT_OFM_NUM / 16;

    uint out_idx = batch_idx % 16 +
        tile_y_idx * output_tile_height * weightsOSplit +
        batch_idx / 16 * weightsOSplit * OUTPUT_SIZE_Y +
        feature_idx * weightsOSplit * OUTPUT_SIZE_Y * oDivSplit +
        tile_x_idx * output_tile_width * weightsOSplit * OUTPUT_SIZE_Y * oDivSplit * INPUT0_IFM_NUM;

    output[out_idx] = TO_OUTPUT_TYPE(+90.0 / 90 * tile.x); out_idx += weightsOSplit * OUTPUT_SIZE_Y * oDivSplit * INPUT0_IFM_NUM;
    output[out_idx] = TO_OUTPUT_TYPE(-20.0 / 90 * tile.x - 20.0 / 90 * tile.y - 20.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y * oDivSplit * INPUT0_IFM_NUM;
    output[out_idx] = TO_OUTPUT_TYPE(-20.0 / 90 * tile.x + 20.0 / 90 * tile.y - 20.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y * oDivSplit * INPUT0_IFM_NUM;
    output[out_idx] = TO_OUTPUT_TYPE(+1.0 / 90 * tile.x + 2.0 / 90 * tile.y + 4.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y * oDivSplit * INPUT0_IFM_NUM;
    output[out_idx] = TO_OUTPUT_TYPE(+1.0 / 90 * tile.x - 2.0 / 90 * tile.y + 4.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y * oDivSplit * INPUT0_IFM_NUM;
    output[out_idx] = TO_OUTPUT_TYPE(+64.0 / 90 * tile.x + 32.0 / 90 * tile.y + 16.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y * oDivSplit * INPUT0_IFM_NUM;
    output[out_idx] = TO_OUTPUT_TYPE(+64.0 / 90 * tile.x - 32.0 / 90 * tile.y + 16.0 / 90 * tile.z); out_idx += weightsOSplit * OUTPUT_SIZE_Y * oDivSplit * INPUT0_IFM_NUM;
    output[out_idx] = TO_OUTPUT_TYPE(+90.0 / 90 * tile.z);
}
