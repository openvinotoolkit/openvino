// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if MAX_POOLING
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

KERNEL(pooling_gpu)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1) * POOL_SIZE_Y;
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;

    if ((x >= OUTPUT_SIZE_X) || (y >= OUTPUT_SIZE_Y))
        return;

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

    uint input_idx = GET_DATA_INDEX(INPUT0, b, f, offset_y, offset_x);
    ACCUMULATOR_TYPE max_x[BLOCK_SIZE_Y];
    ACCUMULATOR_TYPE result[POOL_SIZE_Y];

    for(uint i = 0; i < BLOCK_SIZE_Y; i++)
    {
        max_x[i] = INIT_VAL;
    }

    // we do max in "x" dimension
    for(uint j = 0; j < BLOCK_SIZE_Y; j++)
    {
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            max_x[j] = FUNC_CALL(apply_pooling)(max_x[j], TO_ACCUMULATOR_TYPE(input[input_idx]));
            input_idx += INPUT0_X_PITCH;
        }
        input_idx += (INPUT0_Y_PITCH - POOL_SIZE_X*INPUT0_X_PITCH);
    }

    for(uint i = 0; i < POOL_SIZE_Y; i++)
    {
        result[i] = max_x[i * STRIDE_SIZE_Y];
    }

    // now we do max in "y" dimension
    for(uint i = 0; i < POOL_SIZE_Y; i++)
    {
        for(uint j = 1; j < POOL_SIZE_Y; j++)
        {


            result[i] = FUNC_CALL(apply_pooling)(result[i], max_x[j + i * STRIDE_SIZE_Y]);
        }
    }

    uint output_pos = GET_DATA_INDEX(OUTPUT, b, f, y, x);

    OUTPUT_TYPE final_result;
    ACTIVATION_TYPE pool_result;

    for(uint i = 0; i < POOL_SIZE_Y; i++)
    {
        if((y + i) < OUTPUT_SIZE_Y)
        {
#if defined AVG_POOLING
            result[i] /= TO_ACCUMULATOR_TYPE(POOL_SIZE_Y * POOL_SIZE_X);
#endif
            pool_result = TO_ACTIVATION_TYPE(result[i]);
        #if HAS_FUSED_OPS
            FUSED_OPS;
            final_result = FUSED_OPS_RESULT;
        #else
            final_result = TO_OUTPUT_TYPE(ACTIVATION(pool_result, ACTIVATION_PARAMS));
        #endif
            output[output_pos] = final_result;
            output_pos += OUTPUT_Y_PITCH;
        }
    }
}

#undef INIT_VAL
