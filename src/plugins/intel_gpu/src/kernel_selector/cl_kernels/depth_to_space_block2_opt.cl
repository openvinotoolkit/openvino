// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(depth_to_space_block2_opt)(const __global half* input, __global half* output)
{
    const int in_height  = get_global_size(1);
    const int2 pos = { get_global_id(0), get_global_id(1) };

    if (pos.x >= (IN_WIDTH) || pos.y >= in_height) return;

    const int offset = IN_WIDTH * in_height;

    __attribute__((opencl_unroll_hint(OUTPUT_FEATURE_NUM)))
    for (uint ofm_id=0; ofm_id < OUTPUT_FEATURE_NUM; ofm_id++){
        int add_off = offset * 2 * ofm_id * BLOCK_SIZE * BLOCK_SIZE;
        int ofm_x_offset = offset * ofm_id;
        const int inIdx = IN_WIDTH * pos.y + pos.x + ofm_x_offset;

        half2 conv_out_0 = ACTIVATION(vload2(inIdx+(offset * 0 * OUTPUT_FEATURE_NUM), input ), ACTIVATION_PARAMS);
        half2 conv_out_1 = ACTIVATION(vload2(inIdx+(offset * 1 * OUTPUT_FEATURE_NUM), input ), ACTIVATION_PARAMS);
        half2 conv_out_2 = ACTIVATION(vload2(inIdx+(offset * 2 * OUTPUT_FEATURE_NUM), input ), ACTIVATION_PARAMS);
        half2 conv_out_3 = ACTIVATION(vload2(inIdx+(offset * 3 * OUTPUT_FEATURE_NUM), input ), ACTIVATION_PARAMS);

        int outIdx1 = IN_WIDTH * BLOCK_SIZE * pos.y + pos.x;
        int outIdx2 = outIdx1 + IN_WIDTH;

        vstore2((float2)(as_float((half2)(conv_out_0.s0, conv_out_1.s0)), as_float((half2)(conv_out_0.s1, conv_out_1.s1))), outIdx1, (__global float*) (output + add_off));
        vstore2((float2)(as_float((half2)(conv_out_2.s0, conv_out_3.s0)), as_float((half2)(conv_out_2.s1, conv_out_3.s1))), outIdx2, (__global float*) (output + add_off));
    }
}
