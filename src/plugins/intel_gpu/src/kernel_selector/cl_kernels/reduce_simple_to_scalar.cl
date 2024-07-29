// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

inline uint FUNC(calc_linear_offset)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint v, uint u, uint w, uint z, uint y, uint x)
{
    uint index = b * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W * OUTPUT_SIZE_U * OUTPUT_SIZE_V * OUTPUT_FEATURE_NUM +
                 f * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W * OUTPUT_SIZE_U * OUTPUT_SIZE_V +
                 v * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W * OUTPUT_SIZE_U +
                 u * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z * OUTPUT_SIZE_W +
                 w * OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z +
                 z * OUTPUT_SIZE_X * OUTPUT_SIZE_Y +
                 y * OUTPUT_SIZE_X +
                 x;

    return index;
}

KERNEL(reduce_ref)(
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

    ACCUMULATOR_TYPE acc = 0;
    uint counter = 0;
    for (uint i = bi; i < TOTAL_NUM_ELEMENTS; i += BLOCK_STRIDE) {
#ifdef REDUCE_SUM_MODE
        acc += data[i];
#elif REDUCE_MAX_MODE
        if (counter == 0)
            acc = data[i];
        else
            acc = data[i] > acc ? data[i] : acc;
#elif REDUCE_MIN_MODE
        if (counter == 0)
            acc = data[i];
        else
            acc = data[i] < acc ? data[i] : acc;
#elif REDUCE_MEAN_MODE
        acc += data[i];
#elif REDUCE_PROD_MODE
        if (counter == 0)
            acc = data[i];
        else
            acc *= data[i];
#endif
        counter++;
    }

    buffer[bi] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (bi != 0)
        return;
    
    acc = 0;
    counter = 0;
    for (uint i = 0; i < NUM_BLOCKS; i++) {
#ifdef REDUCE_SUM_MODE
        acc += buffer[i];
#elif REDUCE_MAX_MODE
        if (counter == 0)
            acc = buffer[i];
        else
            acc = buffer[i] > acc ? buffer[i] : acc;
#elif REDUCE_MIN_MODE
        if (counter == 0)
            acc = buffer[i];
        else
            acc = buffer[i] < acc ? buffer[i] : acc;
#elif REDUCE_MEAN_MODE
        acc += buffer[i];
#elif REDUCE_PROD_MODE
        if (counter == 0)
            acc = buffer[i];
        else
            acc *= buffer[i];
#endif
        counter++;
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
