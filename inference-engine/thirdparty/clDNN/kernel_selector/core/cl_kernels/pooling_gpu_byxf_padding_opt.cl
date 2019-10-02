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

#define VECTOR_TYPE MAKE_VECTOR_TYPE(UNIT_TYPE,8)
#define FEATURE_PER_ITEM 8
#define FEATURE_BLOCK_NUM (OUTPUT_FEATURE_NUM / 8)

#if   defined MAX_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_MIN
#elif defined AVG_POOLING
    #define UNIT_INIT_VAL UNIT_VAL_ZERO
#else
#error
#endif

inline VECTOR_TYPE FUNC(apply_pooling)(VECTOR_TYPE tmp, VECTOR_TYPE in)
{
#if   defined MAX_POOLING
    return max(tmp, in);
#elif defined AVG_POOLING
    return tmp + in;
#endif
}

KERNEL(pooling_gpu_byxf_opt)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    VECTOR_TYPE out;
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf / INPUT0_BATCH_NUM * FEATURE_PER_ITEM;
    const uint b    = bf % INPUT0_BATCH_NUM;
    
    VECTOR_TYPE feature_block;
    
    if ((x >= OUTPUT_SIZE_X) || (y >= OUTPUT_SIZE_Y))
        return;

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

#ifdef CHECK_BOUNDRY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }
#endif
    int input_idx = b*FEATURE_BLOCK_NUM*INPUT0_SIZE_X*INPUT0_SIZE_Y + FEATURE_BLOCK_NUM*INPUT0_SIZE_X*offset_y + FEATURE_BLOCK_NUM*offset_x + bf / INPUT0_BATCH_NUM;

    out = UNIT_INIT_VAL;

    __attribute__((opencl_unroll_hint))
    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;
        bool zero = input_offset_y < 0 || input_offset_y >= INPUT0_SIZE_Y;
        if(!zero)
        {
            __attribute__((opencl_unroll_hint))
            for(uint i = 0; i < POOL_SIZE_X; i++)
            {
                int input_offset_x = offset_x + i;
                zero = input_offset_x < 0 || input_offset_x >= INPUT0_SIZE_X;
                if (!zero)
                {
                    feature_block = vload8(input_idx+FEATURE_BLOCK_NUM*i, input);
                    out = FUNC_CALL(apply_pooling)(out, feature_block);
                }
            }
        }
        input_idx += FEATURE_BLOCK_NUM*INPUT0_SIZE_X;
    }

    uint output_pos = GET_DATA_INDEX(OUTPUT, b, f, y, x);
    __attribute__((opencl_unroll_hint))
    for(uint i = 0; i < FEATURE_PER_ITEM; i++)
    {
        if(f+i < INPUT0_FEATURE_NUM)
        {
#if defined MAX_POOLING
            output[output_pos+i] = ACTIVATION(out[i], ACTIVATION_PARAMS);
#elif defined AVG_POOLING
            output[output_pos+i] = ACTIVATION(out[i]/(UNIT_TYPE)(POOL_SIZE_X*POOL_SIZE_Y), ACTIVATION_PARAMS);
#endif
        }
    }
}

#undef FEATURE_BLOCK_NUM
#undef FEATURE_PER_ITEM
#undef UNIT_INIT_VAL
#undef VECTOR_TYPE