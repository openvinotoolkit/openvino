// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#define INPUT0_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
#define INPUT1_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT1_TYPE, 8)
#define OUTPUT_VEC_TYPE  MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8)

#define TO_VECTOR_TYPE_IMPL_8(elem_type)  CAT(convert_##elem_type, 8)
#define TO_VECTOR_TYPE(elem_type, size)   CAT(TO_VECTOR_TYPE_IMPL_, size)(elem_type)
#define TO_VECTOR_TYPE_IMPL_SAT_8(elem_type)  CAT(convert_##elem_type, 8##_sat)
#define TO_VECTOR_TYPE_IMPL_SAT_RTE_8(elem_type)  CAT(convert_##elem_type, 8##_sat_rte)
#define TO_VECTOR_TYPE_SAT(elem_type, size)   CAT(TO_VECTOR_TYPE_IMPL_SAT_, size)(elem_type)
#define TO_VECTOR_TYPE_SAT_RTE(elem_type, size)   CAT(TO_VECTOR_TYPE_IMPL_SAT_RTE_, size)(elem_type)
// Convert built-in functions with _sat modifier are not supported in floating point
//create defines without _sat to overcome this issue
#define convert_float8_sat  convert_float8
#define convert_half8_sat   convert_half8

#ifdef SUB_GROUP_SIZE
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
#endif
#ifndef IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
#endif

KERNEL(quantize_gpu_scale_shift_vload8_opt)(OPTIONAL_SHAPE_INFO_ARG
                                     const __global INPUT0_TYPE* input,
                                     const __global INPUT1_TYPE* input_low,
                                     const __global INPUT2_TYPE* input_high,
                                     const __global INPUT3_TYPE* output_low,
                                     const __global INPUT4_TYPE* output_high,
                                     const __global INPUT5_TYPE* input_scale,
                                     const __global INPUT6_TYPE* input_shift,
                                     const __global INPUT7_TYPE* output_scale,
                                     const __global INPUT8_TYPE* output_shift,
                                           __global OUTPUT_TYPE* output)
{
    const int global_id = get_global_id(0);
#if LAST_ACCESSED_X
    global_id -= max(global_id - (int)LAST_ACCESSED_X, 0);
#endif
    const INPUT0_VEC_TYPE in0 = vload8(global_id, input);
    OUTPUT_VEC_TYPE res;
    const INPUT1_VEC_TYPE vscale = IN_SCALE_VAL;
    const INPUT1_VEC_TYPE vshift = IN_SHIFT_VAL;
    const INPUT1_VEC_TYPE ovscale = OUT_SCALE_VAL;
    const INPUT1_VEC_TYPE ovshift = OUT_SHIFT_VAL;
#if HAS_CLAMP
#if CAN_USE_OUTPUT_RANGE
    INPUT1_TYPE output_low_val   = OUT_LO_VAL;
    INPUT1_TYPE output_high_val  = OUT_HI_VAL;
#else
    INPUT1_TYPE input_low_val    = IN_LO_VAL;
    INPUT1_TYPE input_high_val   = IN_HI_VAL;
#endif // CAN_USE_OUTPUT_RANGE
#endif // HAS_CLAMP

// ************************************************************* //
// Calculations for optimized branch with the output range usage //
// ************************************************************* //

#if CAN_USE_OUTPUT_RANGE

#if HAS_PRE_SHIFT
    INPUT1_VEC_TYPE val = TO_VECTOR_TYPE(INPUT1_TYPE, 8)(in0) * vscale + vshift;
#else
    INPUT1_VEC_TYPE val = TO_VECTOR_TYPE(INPUT1_TYPE, 8)(in0) * vscale;
#endif

#if HAS_OUTPUT_RANGE_ROUND
    val = round(val);
#endif

#if HAS_POST_SCALE
    val *= ovscale;
#endif

#if HAS_POST_SHIFT
    val += ovshift;
#endif

#if HAS_CLAMP
#if HAS_MIN_CLAMP && HAS_MAX_CLAMP
    val = clamp(val, output_low_val, output_high_val);
#elif HAS_MIN_CLAMP
    val = max(val, output_low_val);
#else // HAS_MAX_CLAMP
    val = min(val, output_high_val);
#endif
#endif // HAS_CLAMP

// ************************************************************** //
// Calculations for alternative branch with the input range usage //
// ************************************************************** //

#else // CAN_USE_OUTPUT_RANGE

#if HAS_CLAMP
    INPUT1_VEC_TYPE val = clamp(TO_VECTOR_TYPE(INPUT1_TYPE, 8)(in0), input_low_val, input_high_val);
#else
    INPUT1_VEC_TYPE val = TO_VECTOR_TYPE(INPUT1_TYPE, 8)(in0);
#endif

#if HAS_PRE_SHIFT
    val = round(val * vscale + vshift);
#else
    val = round(val * vscale);
#endif

#if HAS_POST_SCALE
    val *= ovscale;
#endif

#if HAS_POST_SHIFT
    val += ovshift;
#endif

#endif // CAN_USE_OUTPUT_RANGE

#if OUTPUT_IS_FP
        res = TO_VECTOR_TYPE_SAT(OUTPUT_TYPE, 8)(val);
#else
        res = TO_VECTOR_TYPE_SAT_RTE(OUTPUT_TYPE, 8)(val);;
#endif
    //BLOCK_WRITEN(OUTPUT_TYPE, 8, output, global_id, res);
    vstore8(res, global_id, output);
}