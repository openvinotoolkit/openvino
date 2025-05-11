// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define INPUT0_VEC8 MAKE_VECTOR_TYPE(INPUT0_TYPE,8)

#define ACCUMULATOR_VEC8 MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 8)
#define TO_ACCUMULATOR_VEC8 CAT(convert_, ACCUMULATOR_VEC8)

#define FEATURE_PER_ITEM 8
#define FEATURE_BLOCK_NUM (INPUT0_FEATURE_NUM / 8)

#if MAX_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_MIN
#elif AVG_POOLING
    #define INIT_VAL ACCUMULATOR_VAL_ZERO
#else
    #error
#endif

inline ACCUMULATOR_VEC8 FUNC(apply_pooling)(ACCUMULATOR_VEC8 tmp, ACCUMULATOR_VEC8 in)
{
#if   defined MAX_POOLING
    return ACCUMULATOR_MAX_FUNC(tmp, in);
#elif defined AVG_POOLING
    return tmp + in;
#endif
}

KERNEL(pooling_gpu_byxf_opt)(
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
    const uint f    = bf / INPUT0_BATCH_NUM * FEATURE_PER_ITEM;
    const uint b    = bf % INPUT0_BATCH_NUM;

    INPUT0_VEC8 feature_block;
    ACCUMULATOR_VEC8 result;

    if ((x >= OUTPUT_SIZE_X) || (y >= OUTPUT_SIZE_Y))
        return;

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

#ifdef CHECK_BOUNDARY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }
#endif
    int input_idx = b*FEATURE_BLOCK_NUM*INPUT0_SIZE_X*INPUT0_SIZE_Y + FEATURE_BLOCK_NUM*INPUT0_SIZE_X*offset_y + FEATURE_BLOCK_NUM*offset_x + bf / INPUT0_BATCH_NUM;

    result = INIT_VAL;

    unroll_for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;
        bool zero = input_offset_y < 0 || input_offset_y >= INPUT0_SIZE_Y;
        if(!zero)
        {
            unroll_for(uint i = 0; i < POOL_SIZE_X; i++)
            {
                int input_offset_x = offset_x + i;
                zero = input_offset_x < 0 || input_offset_x >= INPUT0_SIZE_X;
                if (!zero)
                {
                    feature_block = vload8(input_idx+FEATURE_BLOCK_NUM*i, input);
                    result = FUNC_CALL(apply_pooling)(result, TO_ACCUMULATOR_VEC8(feature_block));
                }
            }
        }
        input_idx += FEATURE_BLOCK_NUM*INPUT0_SIZE_X;
    }

   OUTPUT_TYPE final_result;

    uint output_pos = GET_DATA_INDEX(OUTPUT, b, f, y, x);
    unroll_for(uint i = 0; i < FEATURE_PER_ITEM; i++)
    {
        if(f+i < INPUT0_FEATURE_NUM)
        {
#if defined MAX_POOLING
            ACTIVATION_TYPE pool_result = TO_ACTIVATION_TYPE(result[i]);
        #if HAS_FUSED_OPS
            FUSED_OPS;
            final_result = FUSED_OPS_RESULT;
        #else
            final_result = TO_OUTPUT_TYPE(ACTIVATION(pool_result, ACTIVATION_PARAMS));
        #endif
            output[output_pos+i] = final_result;
#elif defined AVG_POOLING
            ACTIVATION_TYPE pool_result = TO_ACTIVATION_TYPE(result[i]/(OUTPUT_TYPE)(POOL_SIZE_X*POOL_SIZE_Y));
        #if HAS_FUSED_OPS
            FUSED_OPS;
            final_result = FUSED_OPS_RESULT;
        #else
            final_result = TO_OUTPUT_TYPE(ACTIVATION(pool_result, ACTIVATION_PARAMS));
        #endif
           output[output_pos+i] = final_result;
#endif
        }
    }
}

#undef FEATURE_BLOCK_NUM
#undef FEATURE_PER_ITEM

#undef INIT_VAL
#undef INPUT0_VEC8

#undef ACCUMULATOR_VEC8
#undef TO_ACCUMULATOR_VEC8
