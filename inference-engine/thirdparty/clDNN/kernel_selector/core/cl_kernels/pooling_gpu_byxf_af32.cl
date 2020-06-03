// Copyright (c) 2016-2020 Intel Corporation
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

KERNEL(pooling_gpu_byxf_af32)(
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
    const uint f    = 4 * (bf % (aligned32_features / 4));
    const uint b    = bf / (aligned32_features / 4);

    typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) input_t;
    if (x >= OUTPUT_SIZE_X)
    {
        return;
    }

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

    ACCUMULATOR_VEC4 result = INIT_VAL;

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

                    input_t input_data = AS_INPUT_TYPE(intel_sub_group_block_read((const __global uint*)(input + input_idx)));
                    result[0] = FUNC_CALL(apply_pooling)(result[0], TO_ACCUMULATOR_TYPE(input_data[0]));
                    result[1] = FUNC_CALL(apply_pooling)(result[1], TO_ACCUMULATOR_TYPE(input_data[1]));
                    result[2] = FUNC_CALL(apply_pooling)(result[2], TO_ACCUMULATOR_TYPE(input_data[2]));
                    result[3] = FUNC_CALL(apply_pooling)(result[3], TO_ACCUMULATOR_TYPE(input_data[3]));

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
            input_t input_data = AS_INPUT_TYPE(intel_sub_group_block_read((const __global uint*)(input + input_idx)));
            result[0] = FUNC_CALL(apply_pooling)(result[0], TO_ACCUMULATOR_TYPE(input_data[0]));
            result[1] = FUNC_CALL(apply_pooling)(result[1], TO_ACCUMULATOR_TYPE(input_data[1]));
            result[2] = FUNC_CALL(apply_pooling)(result[2], TO_ACCUMULATOR_TYPE(input_data[2]));
            result[3] = FUNC_CALL(apply_pooling)(result[3], TO_ACCUMULATOR_TYPE(input_data[3]));

            input_idx += INPUT0_X_PITCH;
        }
        input_idx += (INPUT0_Y_PITCH - POOL_SIZE_X*INPUT0_X_PITCH);
    }

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elementes = POOL_SIZE_X*POOL_SIZE_Y;
#endif
#endif

#if defined AVG_POOLING
#if ENABLE_ROUND
    int4 not_fused_result;
    for (uint i = 0; i < 4; ++i) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        not_fused_result[i] = convert_int(round(((float)result[i] / max(num_elementes, (uint)1)));
    #else
        not_fused_result[i] = convert_int(round((float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X)));
    #endif
    }
#else  // ENABLE_ROUND
    float4 not_fused_result;
    for (uint i = 0; i < 4; ++i) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        not_fused_result[i] = (float)result[i] / max(num_elementes, (uint)1);
    #else
        not_fused_result[i] = (float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X);
    #endif
    }
#endif  // ENABLE_ROUND
#else  // AVG_POOLING
    float4 not_fused_result = convert_float4(result);
#endif  // AVG_POOLING

    OUTPUT_VEC4 final_result;
#if HAS_FUSED_OPS
    ACTIVATION_VEC4 fused_pool_result = TO_ACTIVATION_VEC4(not_fused_result);
    FUSED_OPS;
    final_result = FUSED_OPS_RESULT;
    for(uint op = 0; op < 4; op++)
    {
        const uint output_pos = GET_DATA_INDEX(OUTPUT, b, f+op, y, x);
        output[output_pos] = final_result[op];
    }
#else
    final_result = TO_OUTPUT_VEC4(not_fused_result);
    for(uint op = 0; op < 4; op++)
    {
        const uint output_pos = GET_DATA_INDEX(OUTPUT, b, f+op, y, x);
        final_result[op] = TO_OUTPUT_TYPE(ACTIVATION(not_fused_result[op], ACTIVATION_PARAMS));
        output[output_pos] = final_result[op];
    }
#endif
}

#undef INIT_VAL
#undef ACCUMULATOR_VEC4

#undef ACTIVATION_VEC4
#undef TO_ACTIVATION_VEC4

#undef OUTPUT_VEC4
#undef TO_OUTPUT_VEC4
