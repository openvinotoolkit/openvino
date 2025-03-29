// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(reduce_simple_to_scalar)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* data,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    __local ACCUMULATOR_TYPE buffer[NUM_BLOCKS];

    const uint bi = (uint)get_global_id(0);

    ACCUMULATOR_TYPE acc = data[bi];

    for (uint i = bi + BLOCK_STRIDE; i < TOTAL_NUM_ELEMENTS; i += BLOCK_STRIDE) {
#ifdef REDUCE_SUM_MODE
        acc += data[i];
#elif REDUCE_MEAN_MODE
        acc += data[i];
#elif REDUCE_MAX_MODE
        acc = data[i] > acc ? data[i] : acc;
#elif REDUCE_MIN_MODE
        acc = data[i] < acc ? data[i] : acc;
#elif REDUCE_PROD_MODE
        acc *= data[i];
#endif
    }

    buffer[bi] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (bi != 0)
        return;
    
    acc = buffer[0];

    for (uint i = 1; i < NUM_BLOCKS; i++) {
#ifdef REDUCE_SUM_MODE
        acc += buffer[i];
#elif REDUCE_MEAN_MODE
        acc += buffer[i];
#elif REDUCE_MAX_MODE
        acc = buffer[i] > acc ? buffer[i] : acc;
#elif REDUCE_MIN_MODE
        acc = buffer[i] < acc ? buffer[i] : acc;
#elif REDUCE_PROD_MODE
        acc *= buffer[i];
#endif
    }

    FINAL_ACCUMULATOR_TYPE final_acc = TO_FINAL_ACCUMULATOR_TYPE(acc);
    #if REDUCE_MEAN_MODE
        final_acc /= TOTAL_NUM_ELEMENTS;
    #endif

    OUTPUT_TYPE final_result;
    ACTIVATION_TYPE reduce_result = TO_ACTIVATION_TYPE(final_acc);
#if HAS_FUSED_OPS
    FUSED_OPS;
    final_result = FUSED_OPS_RESULT;
#else
    final_result = TO_OUTPUT_TYPE(ACTIVATION(reduce_result, ACTIVATION_PARAMS));
#endif
    output[0] = final_result;
}
