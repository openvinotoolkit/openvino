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

__attribute__((intel_reqd_sub_group_size(32)))
KERNEL(pooling_gpu_fs_bs_yx_bsv4_fsv32_simd32)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output)
{
    const uint x    = (uint)get_group_id(0);
    const uint y    = (uint)get_group_id(1);
    const uint bf   = (uint)get_group_id(2) * BATCH_SG_COUNT + get_sub_group_id();
	// we process 4 features per workitem that's why we need to divide it
    const uint aligned32_features = ((INPUT0_FEATURE_NUM + 31) / 32) * 32;
    const uint f = ((bf * 32) % aligned32_features) + (get_sub_group_local_id() % 8) * 4;
    const uint b = 4 * ((bf * 32) / aligned32_features) + (get_sub_group_local_id() / 8);
    
    if (x >= OUTPUT_SIZE_X)
    {
        return;
    }

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    
    int4 result = INIT_VAL;

    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }

    const uint batch_and_feature_offset = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(INPUT0, b, f, 0, 0);
    __attribute__((opencl_unroll_hint(POOL_SIZE_Y)))
    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;

        __attribute__((opencl_unroll_hint(POOL_SIZE_X)))
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            int input_offset_x = offset_x + i;
            bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;
            bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;
            bool zero =  (zero_x || zero_y);
            const uint input_idx =  zero ? 0 : batch_and_feature_offset + input_offset_y*IN_Y_PITCH + input_offset_x*IN_X_PITCH;

            const __global uint* input_uint = (const __global uint*)(input + input_idx);
            int int_data = as_int(input_uint[0]);

            char4 input_data = zero ? (char4)(INIT_VAL,INIT_VAL,INIT_VAL,INIT_VAL) : as_char4(int_data);
            result[0] = FUNC_CALL(apply_pooling)((int)result[0], (int)input_data[0]);
            result[1] = FUNC_CALL(apply_pooling)((int)result[1], (int)input_data[1]);
            result[2] = FUNC_CALL(apply_pooling)((int)result[2], (int)input_data[2]);
            result[3] = FUNC_CALL(apply_pooling)((int)result[3], (int)input_data[3]);
        }
    }

    char4 char_res;
    for(uint op = 0; op < 4; op++)
    {
        char_res[op] = ACTIVATION(convert_char(result[op]), ACTIVATION_PARAMS);
    }

    const uint output_pos = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y, x);

    __global uint* output_uint = (__global uint*)(output + output_pos);
    output_uint[0] = as_uint(char_res);
}

#undef INIT_VAL
