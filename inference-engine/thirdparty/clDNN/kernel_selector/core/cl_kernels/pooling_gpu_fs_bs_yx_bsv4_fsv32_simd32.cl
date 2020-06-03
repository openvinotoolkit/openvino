// Copyright (c) 2020 Intel Corporation
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

#define ACTIVATION_VEC4 MAKE_VECTOR_TYPE(ACTIVATION_TYPE, 4)
#define TO_ACTIVATION_VEC4 CAT(convert_, ACTIVATION_VEC4)

#define ACCUMULATOR_VEC4 MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 4)

#define OUTPUT_VEC4 MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4)
#define TO_OUTPUT_VEC4 CAT(convert_, OUTPUT_VEC4)

#if MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#else
    #error
#endif

inline ACCUMULATOR_TYPE FUNC(apply_pooling)(ACCUMULATOR_TYPE tmp, ACCUMULATOR_TYPE in)
{
#if MAX_POOLING
    return ACCUMULATOR_MAX_FUNC(tmp, in);
#endif
}

__attribute__((intel_reqd_sub_group_size(32)))
KERNEL(pooling_gpu_fs_bs_yx_bsv4_fsv32_simd32)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint x    = (uint)get_group_id(0);
    const uint y    = (uint)get_group_id(1);
    const uint bf   = (uint)get_group_id(2) * BATCH_SG_COUNT + (uint)get_sub_group_id();
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

    ACCUMULATOR_VEC4 result = INIT_VAL;

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
            result[0] = FUNC_CALL(apply_pooling)(result[0], TO_ACCUMULATOR_TYPE(input_data[0]));
            result[1] = FUNC_CALL(apply_pooling)(result[1], TO_ACCUMULATOR_TYPE(input_data[1]));
            result[2] = FUNC_CALL(apply_pooling)(result[2], TO_ACCUMULATOR_TYPE(input_data[2]));
            result[3] = FUNC_CALL(apply_pooling)(result[3], TO_ACCUMULATOR_TYPE(input_data[3]));
        }
    }

    OUTPUT_VEC4 final_result;

    #if HAS_FUSED_OPS
        ACTIVATION_VEC4 pool_result;
        pool_result = TO_ACTIVATION_VEC4(TO_OUTPUT_VEC4(result));
        FUSED_OPS;
        final_result = FUSED_OPS_RESULT;
    #else
        char4 pool_result;
        for(uint op = 0; op < 4; op++)
        {
            pool_result[op] = ACTIVATION(TO_OUTPUT_TYPE(result[op]), ACTIVATION_PARAMS);
        }
        final_result = TO_OUTPUT_VEC4(pool_result);
    #endif

    const uint output_pos = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y, x);
    *((__global OUTPUT_VEC4*)(output + output_pos)) = final_result;
}

#undef INIT_VAL
#undef ACCUMULATOR_VEC4

#undef ACTIVATION_VEC4
#undef TO_ACTIVATION_VEC4

#undef OUTPUT_VEC4
#undef TO_OUTPUT_VEC4
