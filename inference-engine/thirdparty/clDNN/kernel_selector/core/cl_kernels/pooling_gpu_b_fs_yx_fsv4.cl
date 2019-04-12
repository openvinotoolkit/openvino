// Copyright (c) 2019 Intel Corporation
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

KERNEL(pooling_gpu_b_fs_yx_fsv4)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output)
{
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint f    = (bf * 4) % INPUT0_FEATURE_NUM;
    const uint b    = (bf * 4) / INPUT0_FEATURE_NUM;

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

    int result[4] = { INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL };

#ifdef CHECK_BOUNDRY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

    const uint batch_and_feature_offset = GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, 0, 0);
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
                    const uint input_idx = batch_and_feature_offset + input_offset_y*IN_Y_PITCH + input_offset_x*IN_X_PITCH;

                    int int_data   = *((const __global int*)(input + input_idx));
                    char4 ch4_data = as_char4(int_data);
                    result[0] = FUNC_CALL(apply_pooling)(result[0], (int)ch4_data[0]);
                    result[1] = FUNC_CALL(apply_pooling)(result[1], (int)ch4_data[1]);
                    result[2] = FUNC_CALL(apply_pooling)(result[2], (int)ch4_data[2]);
                    result[3] = FUNC_CALL(apply_pooling)(result[3], (int)ch4_data[3]);

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
#else // !CHECK_BOUNDRY
    uint input_idx = GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, offset_y, offset_x);

    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            int int_data   = *((const __global int*)(input + input_idx));
            char4 ch4_data = as_char4(int_data);
            result[0] = FUNC_CALL(apply_pooling)(result[0], (int)ch4_data[0]);
            result[1] = FUNC_CALL(apply_pooling)(result[1], (int)ch4_data[1]);
            result[2] = FUNC_CALL(apply_pooling)(result[2], (int)ch4_data[2]);
            result[3] = FUNC_CALL(apply_pooling)(result[3], (int)ch4_data[3]);

            input_idx += IN_X_PITCH;
        }
        input_idx += (IN_Y_PITCH - POOL_SIZE_X*IN_X_PITCH);
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

    char4 char_res;
    for(uint op = 0; op < 4; op++)
    {
        char_res[op] = ACTIVATION(convert_char(result[op]), NL_M ,NL_N);
    }
    const uint output_pos = GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, b, f, y, x);
    *((__global int*)(output + output_pos)) = as_int(char_res);
}

#undef INIT_VAL
