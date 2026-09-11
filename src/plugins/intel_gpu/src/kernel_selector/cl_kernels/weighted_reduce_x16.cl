// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(weighted_reduce_x16)(const __global INPUT0_TYPE* values, const __global INPUT1_TYPE* weights, __global OUTPUT_TYPE* output) {
    const uint gid = (uint)get_global_id(0);
    const uint y = gid % OUTPUT_SIZE_Y;
    const uint f = (gid / OUTPUT_SIZE_Y) % OUTPUT_FEATURE_NUM;
    const uint b = gid / (OUTPUT_SIZE_Y * OUTPUT_FEATURE_NUM);

    const uint value_idx = INPUT0_GET_INDEX(b, f, y, 0);
    const uint weight_idx = INPUT1_GET_INDEX(b, 0, y, 0);
#if INPUT0_TYPE_SIZE == 2
    volatile ushort8 product_bits0 = as_ushort8(vload8(0, values + value_idx) * vload8(0, weights + weight_idx));
    volatile ushort8 product_bits1 = as_ushort8(vload8(1, values + value_idx) * vload8(1, weights + weight_idx));
    half8 products0 = as_half8(product_bits0);
    half8 products1 = as_half8(product_bits1);
#else
    volatile uint8 product_bits0 = as_uint8(vload8(0, values + value_idx) * vload8(0, weights + weight_idx));
    volatile uint8 product_bits1 = as_uint8(vload8(1, values + value_idx) * vload8(1, weights + weight_idx));
    float8 products0 = as_float8(product_bits0);
    float8 products1 = as_float8(product_bits1);
#endif

#if INPUT0_TYPE_SIZE == 2
    // Match the existing FP16 reduction order after the vload8 Multiply has
    // rounded and stored each product.
    ACCUMULATOR_TYPE acc0 = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE acc1 = ACCUMULATOR_VAL_ZERO;
    unroll_for(uint i = 0; i < 8; ++i) acc0 += TO_ACCUMULATOR_TYPE(products0[i]);
    unroll_for(uint i = 0; i < 8; ++i) acc1 += TO_ACCUMULATOR_TYPE(products1[i]);
    const ACCUMULATOR_TYPE acc = acc0 + acc1;
#else
    // The F32 FSV16 ReduceSum path accumulates both vload8 blocks into one
    // running accumulator.
    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;
    unroll_for(uint i = 0; i < 8; ++i) acc += products0[i];
    unroll_for(uint i = 0; i < 8; ++i) acc += products1[i];
#endif

    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, 0);
    output[output_idx] = TO_OUTPUT_TYPE(acc);
}
