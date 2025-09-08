// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define INPUT_VEC4 MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)

#define ACTIVATION_VEC4 MAKE_VECTOR_TYPE(ACTIVATION_TYPE, 4)
#define TO_ACTIVATION_VEC4 CAT(convert_, ACTIVATION_VEC4)

#define OUTPUT_VEC4 MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4)
#define TO_OUTPUT_VEC4 CAT(convert_, OUTPUT_VEC4)

#if   defined MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif defined AVG_POOLING
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

KERNEL(pooling_gpu_b_fs_yx_fsv4)(
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
    const uint f    = (bf * 4) % ALIGN(INPUT0_FEATURE_NUM, 4);
    const uint b    = (bf * 4) / ALIGN(INPUT0_FEATURE_NUM, 4);

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

    ACCUMULATOR_TYPE result[4] = { INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL };

#ifdef CHECK_BOUNDARY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elements = 0;
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
                    INPUT_VEC4 ch4_data = AS_TYPE(INPUT_VEC4, int_data);
                    result[0] = FUNC_CALL(apply_pooling)(result[0], TO_ACCUMULATOR_TYPE(ch4_data[0]));
                    result[1] = FUNC_CALL(apply_pooling)(result[1], TO_ACCUMULATOR_TYPE(ch4_data[1]));
                    result[2] = FUNC_CALL(apply_pooling)(result[2], TO_ACCUMULATOR_TYPE(ch4_data[2]));
                    result[3] = FUNC_CALL(apply_pooling)(result[3], TO_ACCUMULATOR_TYPE(ch4_data[3]));

#ifdef DYNAMIC_KERNEL_DIVIDER
                    num_elements++;
#endif
                }
            }
        }
    }
#ifdef DYNAMIC_WITH_PADDING_KERNEL_DIVIDER
    const int hend = min(offset_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
    const int wend = min(offset_x + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
    const uint num_elements = (hend - offset_y) * (wend - offset_x);
#endif
#else // !CHECK_BOUNDARY
    uint input_idx = GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, offset_y, offset_x);

    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            int int_data   = *((const __global int*)(input + input_idx));
            INPUT_VEC4 ch4_data = AS_TYPE(INPUT_VEC4, int_data);
            result[0] = FUNC_CALL(apply_pooling)(result[0], TO_ACCUMULATOR_TYPE(ch4_data[0]));
            result[1] = FUNC_CALL(apply_pooling)(result[1], TO_ACCUMULATOR_TYPE(ch4_data[1]));
            result[2] = FUNC_CALL(apply_pooling)(result[2], TO_ACCUMULATOR_TYPE(ch4_data[2]));
            result[3] = FUNC_CALL(apply_pooling)(result[3], TO_ACCUMULATOR_TYPE(ch4_data[3]));

            input_idx += IN_X_PITCH;
        }
        input_idx += (IN_Y_PITCH - POOL_SIZE_X*IN_X_PITCH);
    }

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elements = POOL_SIZE_X*POOL_SIZE_Y;
#endif
#endif

#if defined AVG_POOLING
#if ENABLE_ROUND
    int4 not_fused_result;
    for(uint i = 0; i < 4; i++) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        not_fused_result[i] = convert_int(round(((float)result[i] / max(num_elements, (uint)1))));
    #else
        not_fused_result[i] = convert_int(round((float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X)));
    #endif
    }
#else
    float4 not_fused_result;
    for(uint i = 0; i < 4; i++) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        not_fused_result[i] = (float)result[i] / max(num_elements, (uint)1);
    #else
        not_fused_result[i] = (float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X);
    #endif
    }
#endif  // ENABLE_ROUND
#else  // AVG_POOLING
    int4 not_fused_result;
    for (uint i = 0; i < 4; ++i) {
        not_fused_result[i] = result[i];
    }
#endif  // AVG_POOLING

    ACTIVATION_VEC4 pool_result = TO_ACTIVATION_VEC4(not_fused_result);

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_VEC4 final_result = FUSED_OPS_RESULT;
#else
    OUTPUT_VEC4 final_result = TO_OUTPUT_VEC4(pool_result);
    for(uint op = 0; op < 4; op++)
    {
        final_result[op] = ACTIVATION(final_result[op], ACTIVATION_PARAMS);
    }
#endif

#if OUTPUT_LAYOUT_B_FS_YX_FSV4
    const uint output_pos = OUTPUT_GET_INDEX(b, f, y, x);
#if OUTPUT_FEATURE_NUM % 4 == 0
    *((__global OUTPUT_VEC4*)(output + output_pos)) = final_result;
#else
    for (uint i = 0; i < 4; ++i) {
        if (f + i < OUTPUT_FEATURE_NUM) {
            output[output_pos + i] = final_result[i];
        }
    }
#endif
#else
    for (uint i = 0; i < 4; ++i) {
        if (OUTPUT_FEATURE_NUM % 4 == 0 || f + i < OUTPUT_FEATURE_NUM) {
            output[OUTPUT_GET_INDEX(b, f + i, y, x)] = final_result[i];
        }
    }
#endif
}

#undef INIT_VAL
#undef INPUT_VEC4

#undef ACTIVATION_VEC4
#undef TO_ACTIVATION_VEC4

#undef OUTPUT_VEC4
#undef TO_OUTPUT_VEC4
