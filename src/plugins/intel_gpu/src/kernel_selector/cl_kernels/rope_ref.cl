// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

#ifdef CHATGLM
KERNEL(rope_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* cos_sin,
    __global OUTPUT_TYPE* output)
{
    const uint p = get_global_id(0);
    const uint b = get_global_id(1);
    const uint h = (uint)get_global_id(2) % HEAD_COUNT;
    const uint rf = (uint)get_global_id(2) / HEAD_COUNT;
    uint r = rf < HALF_ROTARY_NDIMS ? rf * 2 : 0;
    uint f = rf < HEAD_SIZE - ROTARY_NDIMS ? rf : 0;

#ifdef ENABLE_SLICE
    uint input_idx = GET_DATA_INDEX(SLICED_INPUT0, p, b, h * HEAD_SIZE, 0);

    input_idx += SLICED_FROM_START * (p * INPUT0_FEATURE_NUM + b + 1)
              + SLICED_FROM_END * (p * INPUT0_FEATURE_NUM + b);
#else
    uint input_idx = INPUT0_GET_INDEX(p, b, h * HEAD_SIZE, 0);
#endif
    uint cos_sin_p = p < INPUT1_BATCH_NUM ? p : 0;
    uint cos_sin_b = b < INPUT1_FEATURE_NUM ? b : 0;
    uint cos_sin_idx = INPUT1_GET_INDEX(cos_sin_p, cos_sin_b, 0, 0);

    uint output_idx = OUTPUT_GET_INDEX(p, b, h, 0);

    INPUT1_TYPE cosv = cos_sin[cos_sin_idx + r];
    INPUT1_TYPE sinv = cos_sin[cos_sin_idx + r + 1];

    INPUT0_TYPE in1 = input[input_idx + r];
    INPUT0_TYPE in2 = input[input_idx + r + 1];

    output[output_idx + r] = cosv * in1 - sinv * in2;
    output[output_idx + r + 1] = sinv * in1 + cosv * in2;

#ifdef ENABLE_IO_COPY
    output[output_idx + ROTARY_NDIMS + f] = input[input_idx + ROTARY_NDIMS + f];
#endif
}
#endif

#ifdef QWEN
KERNEL(rope_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* cos,
    const __global INPUT2_TYPE* sin,
    __global OUTPUT_TYPE* output)
{
    const uint b = get_global_id(0);
    const uint p = get_global_id(1);
    const uint h = (uint)get_global_id(2) / HALF_ROTARY_NDIMS;
    const uint r = (uint)get_global_id(2) % HALF_ROTARY_NDIMS;

#ifdef ENABLE_SLICE
    uint input_idx = GET_DATA_INDEX(SLICED_INPUT0, b, p, h * HEAD_SIZE, 0);

    input_idx += SLICED_FROM_START * (b * INPUT0_FEATURE_NUM + p + 1)
              + SLICED_FROM_END * (b * INPUT0_FEATURE_NUM + p);
#else
    uint input_idx = INPUT0_GET_INDEX(b, p, h * HEAD_SIZE, 0);
#endif
    uint cos_sin_b = b < INPUT1_BATCH_NUM ? b : 0;
    uint cos_sin_p = p + INPUT1_FEATURE_NUM - INPUT0_FEATURE_NUM < INPUT1_FEATURE_NUM ? p + INPUT1_FEATURE_NUM - INPUT0_FEATURE_NUM : 0;
    uint cos_sin_h = h < INPUT1_SIZE_Y ? h : 0;

#ifndef SIN_COS_HAVE_DYNAMIC_PADDINGS
    uint cos_sin_idx = INPUT1_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);

    uint cos_idx = cos_sin_idx;
    uint sin_idx = cos_sin_idx;
#else
    uint cos_idx = INPUT1_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);
    uint sin_idx = INPUT2_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);
#endif

    uint output_idx = OUTPUT_GET_INDEX(b, p, h, 0);

    INPUT0_TYPE in1 = input[input_idx + r];
    INPUT0_TYPE in2 = input[input_idx + HALF_ROTARY_NDIMS + r];

    output[output_idx + r] = cos[cos_idx + r] * in1 - sin[sin_idx + r] * in2;

    output[output_idx + HALF_ROTARY_NDIMS + r] = cos[cos_idx + HALF_ROTARY_NDIMS + r] * in2 +
                                                 sin[sin_idx + HALF_ROTARY_NDIMS + r] * in1;
}
#endif

#ifdef RotateHalf
KERNEL(rope_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* cos,
    const __global INPUT2_TYPE* sin,
#ifdef ENABLE_GATHER
    const __global INPUT3_TYPE* gather,
#endif
    __global OUTPUT_TYPE* output)
{
    const uint b = get_global_id(0);
    const uint h = get_global_id(1);
    const uint p = (uint)get_global_id(2) / HALF_ROTARY_NDIMS;
    const uint r = (uint)get_global_id(2) % HALF_ROTARY_NDIMS;

#ifdef ENABLE_SLICE
    uint input_idx = GET_DATA_INDEX(SLICED_INPUT0, b, h, p, 0);

    input_idx += SLICED_FROM_START * (b * INPUT0_FEATURE_NUM + h + 1)
              + SLICED_FROM_END * (b * INPUT0_FEATURE_NUM + h);
#elif ENABLE_TRANSPOSE
    uint input_idx = GET_DATA_INDEX(TRANSPOSED_INPUT0, b, h, p, 0);
#else
    uint input_idx = INPUT0_GET_INDEX(b, h, p, 0);
#endif

    uint cos_sin_b = b < INPUT1_BATCH_NUM ? b : 0;
    uint cos_sin_h = h < INPUT1_FEATURE_NUM ? h : 0;
    uint cos_sin_p = p;
#ifdef ENABLE_GATHER
    uint gather_b = b < INPUT3_BATCH_NUM ? b : 0;
#if GATHER_RANK == 4
    uint gather_h = h < INPUT3_FEATURE_NUM ? h : 0;
    uint gather_p = p < INPUT3_SIZE_Y ? p : 0;
    uint gather_idx = INPUT3_GET_INDEX(gather_b, gather_h, gather_p, 0);
#else
    uint gather_p = p < INPUT3_FEATURE_NUM ? p : 0;
    uint gather_idx = INPUT3_GET_INDEX(gather_b, gather_p, 0, 0);
#endif
    cos_sin_p = gather[gather_idx];
#endif
    cos_sin_p = cos_sin_p < INPUT1_SIZE_Y ? cos_sin_p : 0;

#ifndef SIN_COS_HAVE_DYNAMIC_PADDINGS
    uint cos_sin_idx = INPUT1_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);

    uint cos_idx = cos_sin_idx;
    uint sin_idx = cos_sin_idx;
#else
    uint cos_idx = INPUT1_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);
    uint sin_idx = INPUT2_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);
#endif

    uint output_idx = OUTPUT_GET_INDEX(b, h, p, 0);

    INPUT0_TYPE in1 = input[input_idx + r];
    INPUT0_TYPE in2 = input[input_idx + HALF_ROTARY_NDIMS + r];

    output[output_idx + r] = cos[cos_idx + r] * in1 - sin[sin_idx + r] * in2;

    output[output_idx + HALF_ROTARY_NDIMS + r] = cos[cos_idx + HALF_ROTARY_NDIMS + r] * in2 +
                                                 sin[sin_idx + HALF_ROTARY_NDIMS + r] * in1;
}
#endif
