// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define TO_OUTPUT_TYPE              CAT(convert_, OUTPUT_TYPE)
#define TO_OUTPUT_TYPE_SAT_RTE      CAT(TO_OUTPUT_TYPE, _sat_rte)

#ifdef SUB_GROUP_SIZE
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
#endif
#ifndef IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
#endif
KERNEL(quantize_gpu_scale_shift_opt)(OPTIONAL_SHAPE_INFO_ARG
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
    const int b = get_global_id(GWS_BATCH);
    const int of = get_global_id(GWS_FEATURE);

#if OUTPUT_DIMS <= 4
    const int yx = get_global_id(GWS_YX);

    const int x = yx % OUTPUT_SIZE_X;
    const int y = yx / OUTPUT_SIZE_X;
    const int z = 0;

    const int output_offset = OUTPUT_GET_INDEX(b, of, y, x);
#elif OUTPUT_DIMS == 5
    const int zyx = get_global_id(GWS_YX);
    const int zyx_div_x = zyx / OUTPUT_SIZE_X;

    const int x = zyx % OUTPUT_SIZE_X;
    const int y = zyx_div_x % OUTPUT_SIZE_Y;
    const int z = zyx_div_x / OUTPUT_SIZE_Y;

    const int output_offset = OUTPUT_GET_INDEX(b, of, z, y, x);
#elif OUTPUT_DIMS == 6
    const int wzyx = get_global_id(GWS_YX);
    const int wzyx_div_x = wzyx / OUTPUT_SIZE_X;
    const int wzyx_div_xy = wzyx_div_x / OUTPUT_SIZE_Y;

    const int x = wzyx % OUTPUT_SIZE_X;
    const int y = wzyx_div_x % OUTPUT_SIZE_Y;
    const int z = wzyx_div_xy % OUTPUT_SIZE_Z;
    const int w = wzyx_div_xy / OUTPUT_SIZE_Z;

    const int output_offset = OUTPUT_GET_INDEX(b, of, w, z, y, x);
#elif OUTPUT_DIMS == 7
    const int uwzyx = get_global_id(GWS_YX);

    const int x = uwzyx % OUTPUT_SIZE_X;
    const int y = uwzyx / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    const int z = uwzyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y % OUTPUT_SIZE_Z;
    const int w = uwzyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z % OUTPUT_SIZE_W;
    const int u = uwzyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z / OUTPUT_SIZE_W;

    const int output_offset = OUTPUT_GET_INDEX(b, of, u, w, z, y, x);
#elif OUTPUT_DIMS == 8
    const int vuwzyx = get_global_id(GWS_YX);

    const int x = vuwzyx % OUTPUT_SIZE_X;
    const int y = vuwzyx / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    const int z = vuwzyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y % OUTPUT_SIZE_Z;
    const int w = vuwzyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z % OUTPUT_SIZE_W;
    const int u = vuwzyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z / OUTPUT_SIZE_W % OUTPUT_SIZE_U;
    const int v = vuwzyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_SIZE_Z / OUTPUT_SIZE_W / OUTPUT_SIZE_U;

    const int output_offset = OUTPUT_GET_INDEX(b, of, v, u, w, z, y, x);
#else
#   error quantize_gpu_scale_shift_opt.cl: output tensors with more than 6 dimensions are unsupported
#endif

#if INPUT0_DIMS <= 4
    const int input_offset = INPUT0_GET_INDEX(b, of, y, x);
#elif INPUT0_DIMS == 5
    const int input_offset = INPUT0_GET_INDEX(b, of, z, y, x);
#elif INPUT0_DIMS == 6
    const int input_offset = INPUT0_GET_INDEX(b, of, w, z, y, x);
#elif INPUT0_DIMS == 7
    const int input_offset = INPUT0_GET_INDEX(b, of, u, w, z, y, x);
#elif INPUT0_DIMS == 8
    const int input_offset = INPUT0_GET_INDEX(b, of, v, u, w, z, y, x);
#else
#   error quantize_gpu_scale_shift_opt.cl: input tensors with more than 6 dimensions are unsupported
#endif

#if HAS_CLAMP && !PER_TENSOR_INPUT_RANGE && !CAN_USE_OUTPUT_RANGE
#if INPUT1_DIMS == 4
    const int in_range_offset = INPUT1_GET_INDEX_SAFE(b, of, y, x);
#elif INPUT1_DIMS == 5
    const int in_range_offset = INPUT1_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT1_DIMS == 6
    const int in_range_offset = INPUT1_GET_INDEX_SAFE(b, of, w, z, y, x);
#elif INPUT1_DIMS == 7
    const int in_range_offset = INPUT1_GET_INDEX_SAFE(b, of, u, w, z, y, x);
#elif INPUT1_DIMS == 8
    const int in_range_offset = INPUT1_GET_INDEX_SAFE(b, of, v, u, w, z, y, x);
#else
#   error quantize_gpu_scale_shift_opt.cl: unsupported INPUT1_DIMS size
#endif
#endif // HAS_CLAMP && !PER_TENSOR_INPUT_RANGE && !CAN_USE_OUTPUT_RANGE

#if INPUT7_DIMS == 4
    const int scales_offset = INPUT7_GET_INDEX_SAFE(b, of, y, x);
#elif INPUT7_DIMS == 5
    const int scales_offset = INPUT7_GET_INDEX_SAFE(b, of, z, y, x);
#elif INPUT7_DIMS == 6
    const int scales_offset = INPUT7_GET_INDEX_SAFE(b, of, w, z, y, x);
#elif INPUT7_DIMS == 7
    const int scales_offset = INPUT7_GET_INDEX_SAFE(b, of, u, w, z, y, x);
#elif INPUT7_DIMS == 8
    const int scales_offset = INPUT7_GET_INDEX_SAFE(b, of, v, u, w, z, y, x);
#else
#   error quantize_gpu_scale_shift_opt.cl: unsupported INPUT7_DIMS size
#endif

#if PER_TENSOR_INPUT_SCALE
    INPUT1_TYPE input_scale_val  = IN_SCALE_VAL;
#else
    INPUT1_TYPE input_scale_val  = input_scale[scales_offset];
#endif

#if PER_TENSOR_INPUT_SHIFT
    INPUT1_TYPE input_shift_val  = IN_SHIFT_VAL;
#else
    INPUT1_TYPE input_shift_val  = input_shift[scales_offset];
#endif

#if PER_TENSOR_OUTPUT_SCALE
    INPUT1_TYPE output_scale_val = OUT_SCALE_VAL;
#else
    INPUT1_TYPE output_scale_val = output_scale[scales_offset];
#endif

#if PER_TENSOR_OUTPUT_SHIFT
    INPUT1_TYPE output_shift_val = OUT_SHIFT_VAL;
#else
    INPUT1_TYPE output_shift_val = output_shift[scales_offset];
#endif

#if HAS_CLAMP
#if CAN_USE_OUTPUT_RANGE
    INPUT1_TYPE output_low_val   = OUT_LO_VAL;
    INPUT1_TYPE output_high_val  = OUT_HI_VAL;
#else
#if PER_TENSOR_INPUT_RANGE
    INPUT1_TYPE input_low_val    = IN_LO_VAL;
    INPUT1_TYPE input_high_val   = IN_HI_VAL;
#else
    INPUT1_TYPE input_low_val    = input_low[in_range_offset];
    INPUT1_TYPE input_high_val   = input_high[in_range_offset];
#endif // PER_TENSOR_INPUT_RANGE
#endif // CAN_USE_OUTPUT_RANGE
#endif // HAS_CLAMP

// ************************************************************* //
// Calculations for optimized branch with the output range usage //
// ************************************************************* //

#if CAN_USE_OUTPUT_RANGE

#if HAS_PRE_SHIFT
    INPUT1_TYPE val = TO_INPUT1_TYPE(input[input_offset]) * input_scale_val + input_shift_val;
#else
    INPUT1_TYPE val = TO_INPUT1_TYPE(input[input_offset]) * input_scale_val;
#endif

#if HAS_OUTPUT_RANGE_ROUND
    val = round(val);
#endif

#if HAS_POST_SCALE
    val *= output_scale_val;
#endif

#if HAS_POST_SHIFT
    val += output_shift_val;
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
    INPUT1_TYPE val = clamp(TO_INPUT1_TYPE(input[input_offset]), input_low_val, input_high_val);
#else
    INPUT1_TYPE val = TO_INPUT1_TYPE(input[input_offset]);
#endif

#if HAS_PRE_SHIFT
    val = round(val * input_scale_val + input_shift_val);
#else
    val = round(val * input_scale_val);
#endif

#if HAS_POST_SCALE
    val *= output_scale_val;
#endif

#if HAS_POST_SHIFT
    val += output_shift_val;
#endif

#endif // CAN_USE_OUTPUT_RANGE

// *********************************** //
// Common section with results writing //
// *********************************** //

#if FEATURE_BLOCKED_FORMAT
    if (of < OUTPUT_FEATURE_NUM)
#endif
#if OUTPUT_IS_FP
        output[output_offset] = TO_OUTPUT_TYPE_SAT(val);
#else
        output[output_offset] = TO_OUTPUT_TYPE_SAT_RTE(val);
#endif
}

#undef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE_SAT_RTE
