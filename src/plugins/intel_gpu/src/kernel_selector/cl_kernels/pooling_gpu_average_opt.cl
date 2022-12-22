// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL(pooling_gpu_average_opt)(
    const __global INPUT0_TYPE* input,
     __global OUTPUT_TYPE* output
)
{
    int local_id = get_local_id(0);
    int tile_x = get_global_id(0);
    int tile_y = get_global_id(1);
    int channel = get_global_id(2);

    int start_x = tile_x / SUB_GROUP_SIZE * TILE_WIDTH;
    int offset_x = start_x + (tile_x - tile_x / SUB_GROUP_SIZE * SUB_GROUP_SIZE) % TILE_WIDTH;
    int offset = INPUT0_SIZE_Y * INPUT0_SIZE_X * channel;

    int start_y = tile_y * TILE_HEIGHT;
    int end_y = min(INPUT0_SIZE_Y - 1, start_y + TILE_HEIGHT - 1);

    // Read 3 lines of SUB_GROUP_SIZE floats.
    // The 3 lines start one float before the current (to the left) and one line up:
    // For example: SUB_GROUP_SIZE=16
    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    // 0 X 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    // In the diagram above X represents the current work item.

    const __global INPUT0_TYPE* base_addr = input + offset + (start_y * INPUT0_SIZE_X + start_x) - 1;

    float input_buffer[3];
    input_buffer[0] = as_float(intel_sub_group_block_read((const __global uint*)(base_addr - INPUT0_SIZE_X)));
    input_buffer[1] = as_float(intel_sub_group_block_read((const __global uint*)(base_addr)));

    int first = 0;
    int second = 1;
    int third = 2;
    float res, sum, sum_1, sum_2;

    for (int y = start_y; y <= end_y; y++)
    {
        base_addr += INPUT0_SIZE_X;

        input_buffer[third] = as_float(intel_sub_group_block_read((const __global uint*)(base_addr)));

#if INPUT0_SIZE_Y == 1
        sum = input_buffer[second];
#else
        if (y == 0)
        {
            sum = input_buffer[second] + input_buffer[third];
        }
        else if (y == INPUT0_SIZE_Y - 1)
        {
            sum = input_buffer[first] + input_buffer[second];
        }
        else
        {
            sum = input_buffer[first] + input_buffer[second] + input_buffer[third];
        }
#endif

        sum_1 = intel_sub_group_shuffle_down(sum, 0.f, 1);
        sum_2 = intel_sub_group_shuffle_down(sum, 0.f, 2);

#if INPUT0_SIZE_X == 1
        res = sum_1 * ONE_OVER_POOL_SIZE;
#else
        if (offset_x == 0)
        {
            res = (sum_1 + sum_2) * ONE_OVER_POOL_SIZE;
        }
        else if (offset_x == INPUT0_SIZE_X - 1)
        {
            res = (sum + sum_1) * ONE_OVER_POOL_SIZE;
        }
        else
        {
            res = (sum + sum_1 + sum_2) * ONE_OVER_POOL_SIZE;
        }
#endif
        OUTPUT_TYPE final_result;

        if ((local_id < TILE_WIDTH) && (offset_x < INPUT0_SIZE_X))
        {
            final_result = TO_OUTPUT_TYPE(ACTIVATION(res, ACTIVATION_PARAMS));
            output[offset + y * INPUT0_SIZE_X + offset_x] = final_result;
        }

        first = (first + 1) % 3;
        second = (second + 1) % 3;
        third = (third + 1) % 3;
    }

}
