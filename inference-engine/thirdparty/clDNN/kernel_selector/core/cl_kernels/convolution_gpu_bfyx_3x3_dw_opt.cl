// Copyright (c) 2016-2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/include_all.cl"

#if FP16_UNIT_USED == 0
    #define ALIGNED_BLOCK_READ(ptr, offset) as_float(intel_sub_group_block_read((const __global uint*)(ptr) + (offset)))
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL(convolution_gpu_bfyx_3x3_dw_opt)(
    __global UNIT_TYPE* input, 
    __global UNIT_TYPE* output, 
    __global UNIT_TYPE* weights, 
#if BIAS_TERM
    __global UNIT_TYPE* biases,
#endif
    uint split_idx)
{
    const uint local_id = get_local_id(0);
    const uint tile_x = get_global_id(0);
    const uint tile_y = get_global_id(1);
    const uint bf = get_global_id(2);
    const uint f = bf % INPUT0_FEATURE_NUM;
    const uint b = bf / INPUT0_FEATURE_NUM;

    const uint start_x = tile_x / SUB_GROUP_SIZE * TILE_WIDTH;
    const uint offset_x = start_x + (tile_x - tile_x / SUB_GROUP_SIZE * SUB_GROUP_SIZE) % TILE_WIDTH;
    const uint offset = b * INPUT0_BATCH_PITCH + INPUT0_FEATURE_PITCH * f;
    const uint out_offset = b * OUTPUT_BATCH_PITCH + OUTPUT_FEATURE_PITCH * f;

    const int start_y = tile_y * TILE_HEIGHT;
    const int end_y = min(INPUT0_SIZE_Y - 1, start_y + TILE_HEIGHT - 1);
    const uint weight_offset = f * FILTER_IFM_PITCH + local_id;

    // Read 3 lines of SUB_GROUP_SIZE floats.
    // The 3 lines start one float before the current (to the left) and one line up:
    // SUB_GROUP_SIZE=16
    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    // 0 X 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    // In the diagram above X represents the current work item.

    const int input_offset_const = INPUT0_OFFSET + offset + (start_y * INPUT0_Y_PITCH + start_x) - 1;

    const uint base_addr_offset = INPUT0_Y_PITCH;

    UNIT_TYPE input_buffer[3] = { UNIT_VAL_ZERO };
    const int base_offset = -base_addr_offset * UNIT_BYTE_SIZE;

#if FP16_UNIT_USED
    const uint lid = get_sub_group_local_id();
    if(input_offset_const - base_addr_offset >= 0)
        input_buffer[0] = input[input_offset_const - base_addr_offset + lid];
    if(input_offset_const >= 0)
        input_buffer[1] = input[input_offset_const + lid];
#else
    input_buffer[0] = ALIGNED_BLOCK_READ(input, input_offset_const - base_addr_offset);
    input_buffer[1] = ALIGNED_BLOCK_READ(input, input_offset_const);
#endif

    UNIT_TYPE w = weights[weight_offset];

    int first = 0;
    int second = 1;
    int third = 2;
    int input_offset = input_offset_const;

    for (int y = start_y; y <= end_y; y++)
    {
        UNIT_TYPE res = UNIT_VAL_ZERO;
        input_offset += base_addr_offset;

#if FP16_UNIT_USED
        if(input_offset >= 0)
            input_buffer[third] = input[input_offset + lid];
#else
        input_buffer[third] = ALIGNED_BLOCK_READ(input, input_offset);
#endif

        uint kc = 0;
        LOOP(FILTER_SIZE_X, kc,
        {
            res = mad(intel_sub_group_shuffle( w, FILTER_SIZE_Y + kc),intel_sub_group_shuffle( input_buffer[second], local_id + kc),res);
            
            if (y == 0)
            {
            res = mad(intel_sub_group_shuffle( w, 2*FILTER_SIZE_Y + kc),intel_sub_group_shuffle( input_buffer[third], local_id + kc),res);
            }
            else if (y == INPUT0_SIZE_Y - 1)
            {
            res = mad(intel_sub_group_shuffle( w, kc),intel_sub_group_shuffle( input_buffer[first], local_id + kc),res);
            }
            else
            {
            res = mad(intel_sub_group_shuffle( w, kc),intel_sub_group_shuffle( input_buffer[first], local_id + kc),res);
            res = mad(intel_sub_group_shuffle( w, 2*FILTER_SIZE_Y + kc),intel_sub_group_shuffle( input_buffer[third], local_id + kc),res);
            }
        });

#if BIAS_TERM
        res += biases[f];
#endif

        if ((local_id < TILE_WIDTH) && (offset_x < INPUT0_SIZE_X))
        {
            output[OUTPUT_OFFSET + out_offset + y * INPUT0_SIZE_X + offset_x] = ACTIVATION(res, NL_M, NL_N);
        }

        first = (first + 1) % 3;
        second = (second + 1) % 3;
        third = (third + 1) % 3;
    }

}

#undef ALIGNED_BLOCK_READ