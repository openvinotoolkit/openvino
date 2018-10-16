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

#if MAX_POOLING || MAX_WITH_ARGMAX_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_MIN
#elif AVG_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_ZERO
#else
#error
#endif


inline UNIT_TYPE FUNC(apply_pooling)(UNIT_TYPE tmp, UNIT_TYPE in)
{
#if MAX_POOLING || MAX_WITH_ARGMAX_POOLING
    return max(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

KERNEL(pooling_gpu)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output
#if MAX_WITH_ARGMAX_POOLING
, __global float* arg_max
#endif
)
{
#if OUTPUT_LAYOUT_BFYX  || OUTPUT_LAYOUT_BYXF
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;
    
    if (x >= OUTPUT_SIZE_X)
    {
        return;
    }
#elif OUTPUT_LAYOUT_YXFB
    const uint x    = (uint)get_global_id(1);
    const uint y    = (uint)get_global_id(2);
    const uint bf   = (uint)get_global_id(0);
    const uint f    = bf / INPUT0_BATCH_NUM;
    const uint b    = bf % INPUT0_BATCH_NUM;
#endif

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    
    UNIT_TYPE result = UNIT_INIT_VAL;
    
#if MAX_WITH_ARGMAX_POOLING
    uint arg_max_idx = 0;
#endif

#ifdef CHECK_BOUNDRY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

    const uint batch_and_feature_offset = GET_DATA_INDEX(INPUT0, b, f, 0, 0);
    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;
        bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;
        if(!zero_y)
        {
            for(uint i = 0; i < POOL_SIZE_X; i++)
            {
                int input_offset_x = offset_x + i;
                bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;
                if(!zero)
                {
                    const uint input_idx = batch_and_feature_offset + input_offset_y*INPUT0_Y_PITCH + input_offset_x*INPUT0_X_PITCH;

#if MAX_WITH_ARGMAX_POOLING
                    if(input[input_idx] > result)
                    {
                        const uint input_idx_bfyx_no_padding = input_offset_x + INPUT0_SIZE_X * (input_offset_y + INPUT0_SIZE_Y * (f + INPUT0_FEATURE_NUM * b));
                        arg_max_idx = input_idx_bfyx_no_padding;
                    }
#endif
                    result = FUNC_CALL(apply_pooling)(result, input[input_idx]);
                    
#ifdef DYNAMIC_KERNEL_DIVIDER
                    num_elementes++;
#endif
                }
            }
        }
    }
#ifdef DYNAMIC_WITH_PADDING_KERNEL_DIVIDER
    const int hend = min(offset_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
    const int wend = min(offset_x + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
    const uint num_elementes = (hend - offset_y) * (wend - offset_x);
#endif
#else
    uint input_idx = GET_DATA_INDEX(INPUT0, b, f, offset_y, offset_x);

#if MAX_WITH_ARGMAX_POOLING
    uint input_idx_bfyx_no_padding = offset_x + INPUT0_SIZE_X * (offset_y + INPUT0_SIZE_Y * (f + INPUT0_FEATURE_NUM * b));
#endif

    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {

#if MAX_WITH_ARGMAX_POOLING
            if(input[input_idx] > result)
                arg_max_idx = input_idx_bfyx_no_padding;
#endif

            result = FUNC_CALL(apply_pooling)(result, input[input_idx]);
            input_idx += INPUT0_X_PITCH;
#if MAX_WITH_ARGMAX_POOLING
            input_idx_bfyx_no_padding++;
#endif
        }
        input_idx += (INPUT0_Y_PITCH - POOL_SIZE_X*INPUT0_X_PITCH);
#if MAX_WITH_ARGMAX_POOLING
        input_idx_bfyx_no_padding += (INPUT0_SIZE_X - POOL_SIZE_X);
#endif
    }
    
#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elementes = POOL_SIZE_X*POOL_SIZE_Y;
#endif
#endif

#if defined AVG_POOLING
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        result /= (UNIT_TYPE)max(num_elementes, (uint)1);
    #else
        result /= (UNIT_TYPE)(POOL_SIZE_Y * POOL_SIZE_X);
    #endif
#endif

    const uint output_pos = GET_DATA_INDEX(OUTPUT, b, f, y, x);
    output[output_pos] = ACTIVATION(result, NL_M ,NL_N);

#if MAX_WITH_ARGMAX_POOLING
    //INPUT1 macro stands for Argmax
    const uint arg_max_pos = GET_DATA_INDEX(INPUT1, b, f, y, x);
    arg_max[arg_max_pos] = convert_float(arg_max_idx);
#endif

}

#undef UNIT_INIT_VAL