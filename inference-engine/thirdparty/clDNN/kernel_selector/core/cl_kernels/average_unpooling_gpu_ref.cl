// Copyright (c) 2018 Intel Corporation
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

KERNEL(average_unpooling_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
#if OUTPUT_LAYOUT_BFYX  || OUTPUT_LAYOUT_BYXF
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;    
#elif OUTPUT_LAYOUT_YXFB
    const uint x    = (uint)get_global_id(1);
    const uint y    = (uint)get_global_id(2);
    const uint bf   = (uint)get_global_id(0);
    const uint f    = bf / INPUT0_BATCH_NUM;
    const uint b    = bf % INPUT0_BATCH_NUM;
#endif

    if (x >= INPUT0_SIZE_X)
    {
        return;
    }
    
    const uint x_begin = x * STRIDE_SIZE_X;
    const uint y_begin = y * STRIDE_SIZE_Y;
    const uint x_end = min((uint)(x_begin + UNPOOL_SIZE_X), (uint)(OUTPUT_SIZE_X));
    const uint y_end = min((uint)(y_begin + UNPOOL_SIZE_Y), (uint)(OUTPUT_SIZE_Y));

    const uint window_x = x_end - x_begin;
    const uint window_y = y_end - y_begin;

    const uint input_offset = GET_DATA_INDEX(INPUT0, b, f, y, x);
    uint out_index = GET_DATA_INDEX(OUTPUT, b, f, y_begin, x_begin);
    UNIT_TYPE out_val = input[input_offset] / (window_x * window_y);

    for(uint j = 0; j < window_y; j++)
    {
        for(uint i = 0; i < window_x; i++)
        {
            output[out_index] += ACTIVATION(out_val, ACTIVATION_PARAMS);
            out_index += OUTPUT_X_PITCH;
        }
        out_index += OUTPUT_Y_PITCH - window_x * OUTPUT_X_PITCH;
    }
}