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

#define OUTPUT_VEC4 MAKE_VECTOR_TYPE(OUTPUT_TYPE,4)
#define TO_OUTPUT_VEC4 CAT(convert_, OUTPUT_VEC4)

#if MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif AVG_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_ZERO
#else
    #error
#endif

inline ACCUMULATOR_TYPE FUNC(apply_pooling)(ACCUMULATOR_TYPE tmp, ACCUMULATOR_TYPE in)
{
#if MAX_POOLING
    return ACCUMULATOR_MAX_FUNC(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(pooling_gpu_fs_bs_yx_bsv4_fsv32)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
	// we process 4 features per workitem that's why we need to divide it
    const uint aligned32_features = ((INPUT0_FEATURE_NUM + 31) / 32) * 32;
    const uint f    = ((uint)get_global_id(2) * 4) % aligned32_features;
    const uint b    = 4 * (((uint)get_global_id(2) * 4) / aligned32_features);
    if (x >= OUTPUT_SIZE_X)
    {
        return;
    }

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    ACCUMULATOR_VEC4 result[4] = { INIT_VAL };

#ifdef CHECK_BOUNDRY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

    const uint batch_and_feature_offset = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(INPUT0, b, f, 0, 0);
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

                    int4 int_data = as_int4(intel_sub_group_block_read4((const __global uint*)(input + input_idx)));
                    for(uint b = 0; b < 4; b++)
                    {
                        char4 input_data = as_char4(int_data[b]);
                        result[b][0] = FUNC_CALL(apply_pooling)(result[b][0], TO_ACCUMULATOR_TYPE(input_data[0]));
                        result[b][1] = FUNC_CALL(apply_pooling)(result[b][1], TO_ACCUMULATOR_TYPE(input_data[1]));
                        result[b][2] = FUNC_CALL(apply_pooling)(result[b][2], TO_ACCUMULATOR_TYPE(input_data[2]));
                        result[b][3] = FUNC_CALL(apply_pooling)(result[b][3], TO_ACCUMULATOR_TYPE(input_data[3]));
                    }

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
    uint input_idx = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(INPUT0, b, f, offset_y, offset_x);

    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            int4 int_data = as_int4(intel_sub_group_block_read4((const __global uint*)(input + input_idx)));
            for(uint b = 0; b < 4; b++)
            {
                char4 input_data = as_char4(int_data[b]);
                result[b][0] = FUNC_CALL(apply_pooling)(result[b][0], TO_ACCUMULATOR_TYPE(input_data[0]));
                result[b][1] = FUNC_CALL(apply_pooling)(result[b][1], TO_ACCUMULATOR_TYPE(input_data[1]));
                result[b][2] = FUNC_CALL(apply_pooling)(result[b][2], TO_ACCUMULATOR_TYPE(input_data[2]));
                result[b][3] = FUNC_CALL(apply_pooling)(result[b][3], TO_ACCUMULATOR_TYPE(input_data[3]));
            }

            input_idx += IN_X_PITCH;
        }
        input_idx += (IN_Y_PITCH - POOL_SIZE_X*IN_X_PITCH);
    }

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elementes = POOL_SIZE_X*POOL_SIZE_Y;
#endif
#endif

#if defined AVG_POOLING
    #if ENABLE_ROUND
        #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
            for(uint b = 0; b < 4; b++)
            {
                for(uint i = 0; i < 4; i++)
                {
                    result[b][i] = TO_ACCUMULATOR_TYPE(round(((float)result[b][i] / max(num_elementes, (uint)1))));
                }
            }
        #else
            for(uint b = 0; b < 4; b++)
            {
                for(uint i = 0; i < 4; i++)
                {
                    result[b][i] = TO_ACCUMULATOR_TYPE(round((float)result[b][i] / (int)(POOL_SIZE_Y * POOL_SIZE_X)));
                }
            }
        #endif
    #else
        #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
            for(uint b = 0; b < 4; b++)
            {
                for(uint i = 0; i < 4; i++)
                {
                    result[b][i] = TO_ACCUMULATOR_TYPE(((float)result[b][i] / max(num_elementes, (uint)1)));
                }
            }
        #else
            for(uint b = 0; b < 4; b++)
            {
                for(uint i = 0; i < 4; i++)
                {
                    result[b][i] = TO_ACCUMULATOR_TYPE((float)result[b][i] / (int)(POOL_SIZE_Y * POOL_SIZE_X));
                }
            }
        #endif
    #endif  // ENABLE_ROUND
#endif  // AVG_POOLING

#if OUTPUT_TYPE_SIZE == 1
    int4 final_result;

    for(uint bi = 0; bi < 4; bi++)
    {
        #if HAS_FUSED_OPS
            ACTIVATION_VEC4 char_result = TO_ACTIVATION_VEC4(convert_char4(result[bi]));
            FUSED_OPS;
            final_result[bi] = as_int(FUSED_OPS_RESULT);
        #else
            char4 char_result = ACTIVATION(convert_char4(result[bi]), ACTIVATION_PARAMS);
            final_result[bi] = as_int(char_result);
        #endif
    }
    const uint output_pos = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y, x);
    intel_sub_group_block_write4((__global uint*)(output + output_pos), as_uint4(final_result));

#elif OUTPUT_TYPE_SIZE == 2 || OUTPUT_TYPE_SIZE == 4
    OUTPUT_VEC4 final_result;

    for(uint bi = 0; bi < 4; bi++)
    {
    #if HAS_FUSED_OPS
        ACTIVATION_VEC4 char_result = TO_ACTIVATION_VEC4(TO_OUTPUT_VEC4(result[bi]));
        FUSED_OPS;
        final_result = FUSED_OPS_RESULT;
    #else
        char4 char_result = ACTIVATION(TO_OUTPUT_VEC4(result[bi]), ACTIVATION_PARAMS);
        final_result = TO_OUTPUT_VEC4(char_result);
    #endif
        const uint output_pos = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b + bi, f, y, x);
        vstore4(final_result, 0, output + output_pos);
    }
#endif
}

#undef INIT_VAL
#undef ACCUMULATOR_VEC4
#undef ACCUMULATOR_VEC4

#undef ACTIVATION_VEC4
#undef TO_ACTIVATION_VEC4

#undef OUTPUT_VEC4
#undef TO_OUTPUT_VEC4
