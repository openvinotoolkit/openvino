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

#if MAX_POOLING
    #define INIT_VAL CHAR_MIN
#elif AVG_POOLING
    #define INIT_VAL 0
#else
#error
#endif


inline int FUNC(apply_pooling)(int tmp, int in)
{
#if MAX_POOLING
    return max(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

KERNEL(pooling_gpu_byxf_af32)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output)
{
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
	// we process 4 features per workitem that's why we need to divide it
    const uint aligned32_features = ((INPUT0_FEATURE_NUM + 31) / 32) * 32;
    const uint f    = 4 * (bf % (aligned32_features / 4));
    const uint b    = bf / (aligned32_features / 4);
    
    if (x >= OUTPUT_SIZE_X)
    {
        return;
    }

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    
    int4 result = INIT_VAL;

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

                    char4 input_data = as_char4(intel_sub_group_block_read((const __global uint*)(input + input_idx)));
                    result[0] = FUNC_CALL(apply_pooling)(result[0], (int)input_data[0]);
                    result[1] = FUNC_CALL(apply_pooling)(result[1], (int)input_data[1]);
                    result[2] = FUNC_CALL(apply_pooling)(result[2], (int)input_data[2]);
                    result[3] = FUNC_CALL(apply_pooling)(result[3], (int)input_data[3]);
                    
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

    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            char4 input_data = as_char4(intel_sub_group_block_read((const __global uint*)(input + input_idx)));
            result[0] = FUNC_CALL(apply_pooling)(result[0], (int)input_data[0]);
            result[1] = FUNC_CALL(apply_pooling)(result[1], (int)input_data[1]);
            result[2] = FUNC_CALL(apply_pooling)(result[2], (int)input_data[2]);
            result[3] = FUNC_CALL(apply_pooling)(result[3], (int)input_data[3]);

            input_idx += INPUT0_X_PITCH;
        }
        input_idx += (INPUT0_Y_PITCH - POOL_SIZE_X*INPUT0_X_PITCH);
    }
    
#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elementes = POOL_SIZE_X*POOL_SIZE_Y;
#endif
#endif

#if defined AVG_POOLING
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        for(uint i = 0; i < 4; i++)
        {
            result[i] = convert_int(round(((float)result[i] / max(num_elementes, (uint)1)));
        }
    #else
        for(uint i = 0; i < 4; i++)
        {
            result[i] = convert_int(round((float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X)));
        }
    #endif
#endif

for(uint op = 0; op < 4; op++)
{
    const uint output_pos = GET_DATA_INDEX(OUTPUT, b, f+op, y, x);
    output[output_pos] = ACTIVATION(convert_char(result[op]), ACTIVATION_PARAMS);
}

}

#undef INIT_VAL