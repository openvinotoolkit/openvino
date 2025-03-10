// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

#define INPUT_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)

#define UNPACK_FLOAT_VEC_1(outputv, input1, input2) \
    outputv.s0 = convert_float(input1.s0);          \
    outputv.s1 = convert_float(input1.s2);          \
    outputv.s2 = convert_float(input1.s4);          \
    outputv.s3 = convert_float(input1.s6);          \
    outputv.s4 = convert_float(input2.s0);          \
    outputv.s5 = convert_float(input2.s2);          \
    outputv.s6 = convert_float(input2.s4);          \
    outputv.s7 = convert_float(input2.s6);

#define UNPACK_FLOAT_VEC_2(outputv, input1, input2) \
    outputv.s0 = convert_float(input1.s1);          \
    outputv.s1 = convert_float(input1.s3);          \
    outputv.s2 = convert_float(input1.s5);          \
    outputv.s3 = convert_float(input1.s7);          \
    outputv.s4 = convert_float(input2.s1);          \
    outputv.s5 = convert_float(input2.s3);          \
    outputv.s6 = convert_float(input2.s5);          \
    outputv.s7 = convert_float(input2.s7);

#define UNPACK_HALF_VEC_1(outputv, input1)  \
    outputv.s0 = convert_float(input1[0]);  \
    outputv.s1 = convert_float(input1[2]);  \
    outputv.s2 = convert_float(input1[4]);  \
    outputv.s3 = convert_float(input1[6]);  \
    outputv.s4 = convert_float(input1[8]);  \
    outputv.s5 = convert_float(input1[10]); \
    outputv.s6 = convert_float(input1[12]); \
    outputv.s7 = convert_float(input1[14]);

#define UNPACK_HALF_VEC_2(outputv, input1)  \
    outputv.s0 = convert_float(input1[1]);  \
    outputv.s1 = convert_float(input1[3]);  \
    outputv.s2 = convert_float(input1[5]);  \
    outputv.s3 = convert_float(input1[7]);  \
    outputv.s4 = convert_float(input1[9]);  \
    outputv.s5 = convert_float(input1[11]); \
    outputv.s6 = convert_float(input1[13]); \
    outputv.s7 = convert_float(input1[15]);

#define UNPACK_HALF16_VEC_1(outputv, input1, input2) \
    outputv = (half16)(input1[0],                    \
                       input1[2],                    \
                       input1[4],                    \
                       input1[6],                    \
                       input1[8],                    \
                       input1[10],                   \
                       input1[12],                   \
                       input1[14],                   \
                       input2[0],                    \
                       input2[2],                    \
                       input2[4],                    \
                       input2[6],                    \
                       input2[8],                    \
                       input2[10],                   \
                       input2[12],                   \
                       input2[14]);

#define UNPACK_HALF16_VEC_2(outputv, input1, input2) \
    outputv = (half16)(input1[1],                    \
                       input1[3],                    \
                       input1[5],                    \
                       input1[7],                    \
                       input1[9],                    \
                       input1[11],                   \
                       input1[13],                   \
                       input1[15],                   \
                       input2[1],                    \
                       input2[3],                    \
                       input2[5],                    \
                       input2[7],                    \
                       input2[9],                    \
                       input2[11],                   \
                       input2[13],                   \
                       input2[15]);

#define PACK_HALF16_VEC_1(outputv, input1, input2) \
    outputv = (half16)(input1[0],                  \
                       input2[0],                  \
                       input1[1],                  \
                       input2[1],                  \
                       input1[2],                  \
                       input2[2],                  \
                       input1[3],                  \
                       input2[3],                  \
                       input1[4],                  \
                       input2[4],                  \
                       input1[5],                  \
                       input2[5],                  \
                       input1[6],                  \
                       input2[6],                  \
                       input1[7],                  \
                       input2[7]);

#define PACK_HALF16_VEC_2(outputv, input1, input2) \
    outputv = (half16)(input1[8],                  \
                       input2[8],                  \
                       input1[9],                  \
                       input2[9],                  \
                       input1[10],                 \
                       input2[10],                 \
                       input1[11],                 \
                       input2[11],                 \
                       input1[12],                 \
                       input2[12],                 \
                       input1[13],                 \
                       input2[13],                 \
                       input1[14],                 \
                       input2[14],                 \
                       input1[15],                 \
                       input2[15]);

#ifdef CHATGLM
KERNEL(rope_opt)(
    OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* cos_sin,
    __global OUTPUT_TYPE* output) {
#if VEC_SIZE != 1 && VEC_SIZE != 8 && VEC_SIZE != 16
#   error "rope_opt.cl - VEC_SIZE must be one of {1, 8, 16}"
#endif

#ifdef SUPPORT_2D_ROPE
    const uint p = get_global_id(0) / HEAD_COUNT;
    const uint h = get_global_id(0) % HEAD_COUNT;
    const uint b = get_global_id(1);   // sequence length
    const uint rf = get_global_id(2);  // max(HALF_ROTARY_NDIMS, HEAD_SIZE - ROTARY_NDIMS)
    uint output_idx = OUTPUT_GET_INDEX(p, h, b, 0);
#else
    const uint p = get_global_id(0);
    const uint b = get_global_id(1);
    const uint h = (uint)get_global_id(2) % HEAD_COUNT;
    const uint rf = (uint)get_global_id(2) / HEAD_COUNT;
    uint output_idx = OUTPUT_GET_INDEX(p, b, h, 0);
#endif

    uint r = rf < HALF_ROTARY_NDIMS ? rf * 2 * VEC_SIZE : 0;
    uint f = rf < HEAD_SIZE - ROTARY_NDIMS ? rf * 2 * VEC_SIZE : 0;

    uint input_idx = INPUT0_GET_INDEX(p, b, h * HEAD_SIZE, 0);
#ifdef ENABLE_SLICE
    input_idx += SLICED_FROM_START;
#endif

    uint cos_sin_p = p < INPUT1_BATCH_NUM ? p : 0;
    uint cos_sin_b = b < INPUT1_FEATURE_NUM ? b : 0;
    uint cos_sin_idx = INPUT1_GET_INDEX(cos_sin_p, cos_sin_b, 0, 0);

#if VEC_SIZE == 1
    float cosv = convert_float(cos_sin[cos_sin_idx + r]);
    float sinv = convert_float(cos_sin[cos_sin_idx + r + 1]);

    float in1 = convert_float(input[input_idx + r]);
    float in2 = convert_float(input[input_idx + r + 1]);

    output[output_idx + r] = TO_OUTPUT_TYPE(cosv * in1 - sinv * in2);
    output[output_idx + r + 1] = TO_OUTPUT_TYPE(sinv * in1 + cosv * in2);

    #ifdef ENABLE_IO_COPY
        output[output_idx + ROTARY_NDIMS + f] = input[input_idx + ROTARY_NDIMS + f];
        output[output_idx + ROTARY_NDIMS + f + 1] = input[input_idx + ROTARY_NDIMS + f + 1];
    #endif
#elif VEC_SIZE == 8
    INPUT_VEC_TYPE inv1 = *(INPUT_VEC_TYPE*)(input + input_idx + r);
    INPUT_VEC_TYPE inv2 = *(INPUT_VEC_TYPE*)(input + input_idx + r + VEC_SIZE);
    INPUT_VEC_TYPE cossinv1 = *(INPUT_VEC_TYPE*)(cos_sin + cos_sin_idx + r);
    INPUT_VEC_TYPE cossinv2 = *(INPUT_VEC_TYPE*)(cos_sin + cos_sin_idx + r + VEC_SIZE);

    float8 in1, in2, cosv, sinv;
    UNPACK_FLOAT_VEC_1(in1, inv1, inv2);
    UNPACK_FLOAT_VEC_2(in2, inv1, inv2);
    UNPACK_FLOAT_VEC_1(cosv, cossinv1, cossinv2);
    UNPACK_FLOAT_VEC_2(sinv, cossinv1, cossinv2);
    float8 out1 = cosv * in1 - sinv * in2;
    float8 out2 = sinv * in1 + cosv * in2;

    *(float8*)(output + output_idx + r) =
        (float8)(out1.s0, out2.s0, out1.s1, out2.s1, out1.s2, out2.s2, out1.s3, out2.s3);
    *(float8*)(output + output_idx + r + VEC_SIZE) =
        (float8)(out1.s4, out2.s4, out1.s5, out2.s5, out1.s6, out2.s6, out1.s7, out2.s7);

    #ifdef ENABLE_IO_COPY
        *(float8*)(output + output_idx + ROTARY_NDIMS + f) = *(float8*)(input + input_idx + ROTARY_NDIMS + f);
        *(float8*)(output + output_idx + ROTARY_NDIMS + f + VEC_SIZE) =
        *(float8*)(input + input_idx + ROTARY_NDIMS + f + VEC_SIZE);
    #endif
#elif VEC_SIZE == 16
    unroll_for(int i = 0; i < 2; i += 1) {
        INPUT_VEC_TYPE inv = *(INPUT_VEC_TYPE*)(input + input_idx + r + i * VEC_SIZE);
        INPUT_VEC_TYPE cossinv = *(INPUT_VEC_TYPE*)(cos_sin + cos_sin_idx + r + i * VEC_SIZE);
        float8 in1, in2, cosv, sinv;
        UNPACK_HALF_VEC_1(in1, inv);
        UNPACK_HALF_VEC_2(in2, inv);
        UNPACK_HALF_VEC_1(cosv, cossinv);
        UNPACK_HALF_VEC_2(sinv, cossinv);
        float8 out1 = cosv * in1 - sinv * in2;
        float8 out2 = sinv * in1 + cosv * in2;

        unroll_for(int j = 0; j < 8; j += 1) {
            output[output_idx + r + i * VEC_SIZE + 2 * j] = TO_OUTPUT_TYPE(out1[j]);
            output[output_idx + r + i * VEC_SIZE + 2 * j + 1] = TO_OUTPUT_TYPE(out2[j]);
        }
    }
    #ifdef ENABLE_IO_COPY
        *(float8*)(output + output_idx + ROTARY_NDIMS + f) = *(float8*)(input + input_idx + ROTARY_NDIMS + f);
        *(float8*)(output + output_idx + ROTARY_NDIMS + f + VEC_SIZE) =
            *(float8*)(input + input_idx + ROTARY_NDIMS + f + VEC_SIZE);
    #endif
#endif
}
#endif

#ifdef QWEN
KERNEL(rope_opt)(
    OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* cos,
    const __global INPUT2_TYPE* sin,
#ifdef ENABLE_GATHER
    const __global INPUT3_TYPE* gather,
#endif
    __global OUTPUT_TYPE* output) {
    const uint b = get_global_id(0);
    const uint p = get_global_id(1);
    const uint h = (uint)get_global_id(2) * VEC_SIZE / HALF_ROTARY_NDIMS;
    const uint r = ((uint)get_global_id(2) * VEC_SIZE) % HALF_ROTARY_NDIMS;

    uint input_idx = INPUT0_GET_INDEX(b, p, h * HEAD_SIZE, 0);
#ifdef ENABLE_SLICE
    input_idx += SLICED_FROM_START;
#endif

    uint cos_sin_b = b < INPUT1_BATCH_NUM ? b : 0;
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
    uint cos_sin_p = gather[gather_idx];
#else
    uint cos_sin_p = p + INPUT1_FEATURE_NUM - INPUT0_FEATURE_NUM < INPUT1_FEATURE_NUM
                         ? p + INPUT1_FEATURE_NUM - INPUT0_FEATURE_NUM
                         : 0;
#endif
    uint cos_sin_h = h < INPUT1_SIZE_Y ? h : 0;

#ifndef SIN_COS_HAVE_DYNAMIC_PADDINGS
    uint cos_sin_idx = INPUT1_GET_INDEX(cos_sin_b, cos_sin_p, cos_sin_h, 0);

    uint cos_idx = cos_sin_idx;
    uint sin_idx = cos_sin_idx;
#else
    uint cos_idx = INPUT1_GET_INDEX(cos_sin_b, cos_sin_p, cos_sin_h, 0);
    uint sin_idx = INPUT2_GET_INDEX(cos_sin_b, cos_sin_p, cos_sin_h, 0);
#endif

    uint output_idx = OUTPUT_GET_INDEX(b, p, h, 0);

#if VEC_SIZE == 1
    INPUT0_TYPE in1 = input[input_idx + r];
    INPUT0_TYPE in2 = input[input_idx + HALF_ROTARY_NDIMS + r];

    output[output_idx + r] = cos[cos_idx + r] * in1 - sin[sin_idx + r] * in2;

    output[output_idx + HALF_ROTARY_NDIMS + r] =
        cos[cos_idx + HALF_ROTARY_NDIMS + r] * in2 + sin[sin_idx + HALF_ROTARY_NDIMS + r] * in1;
#else
    INPUT_VEC_TYPE in1 = *(INPUT_VEC_TYPE*)(input + input_idx + r);
    INPUT_VEC_TYPE in2 = *(INPUT_VEC_TYPE*)(input + input_idx + HALF_ROTARY_NDIMS + r);
    INPUT_VEC_TYPE cos1 = *(INPUT_VEC_TYPE*)(cos + cos_idx + r);
    INPUT_VEC_TYPE cos2 = *(INPUT_VEC_TYPE*)(cos + cos_idx + HALF_ROTARY_NDIMS + r);
    INPUT_VEC_TYPE sin1 = *(INPUT_VEC_TYPE*)(sin + sin_idx + r);
    INPUT_VEC_TYPE sin2 = *(INPUT_VEC_TYPE*)(sin + sin_idx + HALF_ROTARY_NDIMS + r);

    OUTPUT_VEC_TYPE out1 = cos1 * in1 - sin1 * in2;
    OUTPUT_VEC_TYPE out2 = cos2 * in2 + sin2 * in1;

    *(OUTPUT_VEC_TYPE*)(output + output_idx + r) = out1;
    *(OUTPUT_VEC_TYPE*)(output + output_idx + HALF_ROTARY_NDIMS + r) = out2;
#endif
}
#endif

#ifdef RotateHalf
KERNEL(rope_opt)
(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
 const __global INPUT1_TYPE* cos,
 const __global INPUT2_TYPE* sin,
#ifdef ENABLE_GATHER
 const __global INPUT3_TYPE* gather,
#endif
 __global OUTPUT_TYPE* output) {
    const uint b = get_global_id(0);
    const uint h = get_global_id(1);
    const uint p = ((uint)get_global_id(2) * VEC_SIZE) / HALF_ROTARY_NDIMS;
    const uint r = ((uint)get_global_id(2) * VEC_SIZE) % HALF_ROTARY_NDIMS;

#if ENABLE_TRANSPOSE
    uint input_idx = INPUT0_GET_INDEX(b, p, h, 0);
#else
    uint input_idx = INPUT0_GET_INDEX(b, h, p, 0);
    #ifdef ENABLE_SLICE
        input_idx += SLICED_FROM_START;
    #endif
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

#if VEC_SIZE == 1
    INPUT0_TYPE in1 = input[input_idx + r];
    INPUT0_TYPE in2 = input[input_idx + HALF_ROTARY_NDIMS + r];

    output[output_idx + r] = cos[cos_idx + r] * in1 - sin[sin_idx + r] * in2;

    output[output_idx + HALF_ROTARY_NDIMS + r] =
        cos[cos_idx + HALF_ROTARY_NDIMS + r] * in2 + sin[sin_idx + HALF_ROTARY_NDIMS + r] * in1;
#else
    INPUT_VEC_TYPE in1 = *(INPUT_VEC_TYPE*)(input + input_idx + r);
    INPUT_VEC_TYPE in2 = *(INPUT_VEC_TYPE*)(input + input_idx + HALF_ROTARY_NDIMS + r);
    INPUT_VEC_TYPE cos1 = *(INPUT_VEC_TYPE*)(cos + cos_idx + r);
    INPUT_VEC_TYPE cos2 = *(INPUT_VEC_TYPE*)(cos + cos_idx + HALF_ROTARY_NDIMS + r);
    INPUT_VEC_TYPE sin1 = *(INPUT_VEC_TYPE*)(sin + sin_idx + r);
    INPUT_VEC_TYPE sin2 = *(INPUT_VEC_TYPE*)(sin + sin_idx + HALF_ROTARY_NDIMS + r);

    OUTPUT_VEC_TYPE out1 = cos1 * in1 - sin1 * in2;
    OUTPUT_VEC_TYPE out2 = cos2 * in2 + sin2 * in1;

    *(OUTPUT_VEC_TYPE*)(output + output_idx + r) = out1;
    *(OUTPUT_VEC_TYPE*)(output + output_idx + HALF_ROTARY_NDIMS + r) = out2;
#endif
}
#endif

#ifdef RotateInterleaved
KERNEL(rope_opt)(
    OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* cos,
    const __global INPUT2_TYPE* sin,
    __global OUTPUT_TYPE* output) {
#if VEC_SIZE != 1 && VEC_SIZE != 8 && VEC_SIZE != 16
#   error "rope_opt.cl - VEC_SIZE must be one of {1, 8, 16}"
#endif
    const uint b = get_global_id(0);
    const uint h = get_global_id(1);
    const uint p = ((uint)get_global_id(2) * VEC_SIZE) / HALF_ROTARY_NDIMS;
    const uint r = 2 * (((uint)get_global_id(2) * VEC_SIZE) % HALF_ROTARY_NDIMS);

    uint input_idx = INPUT0_GET_INDEX(b, h, p, 0);

    uint cos_sin_b = b < INPUT1_BATCH_NUM ? b : 0;
    uint cos_sin_h = h < INPUT1_FEATURE_NUM ? h : 0;
    uint cos_sin_p = p < INPUT1_SIZE_Y ? p : 0;

#ifndef SIN_COS_HAVE_DYNAMIC_PADDINGS
    uint cos_sin_idx = INPUT1_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);

    uint cos_idx = cos_sin_idx;
    uint sin_idx = cos_sin_idx;
#else
    uint cos_idx = INPUT1_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);
    uint sin_idx = INPUT2_GET_INDEX(cos_sin_b, cos_sin_h, cos_sin_p, 0);
#endif

    uint output_idx = OUTPUT_GET_INDEX(b, h, p, 0);

#if VEC_SIZE == 1
    INPUT0_TYPE in1 = input[input_idx + r];
    INPUT0_TYPE in2 = input[input_idx + r + 1];

    output[output_idx + r] = cos[cos_idx + r] * in1 - sin[sin_idx + r] * in2;
    output[output_idx + r + 1] = cos[cos_idx + r + 1] * in2 + sin[sin_idx + r + 1] * in1;
#elif VEC_SIZE == 8
    INPUT_VEC_TYPE inv1 = *(INPUT_VEC_TYPE*)(input + input_idx + r);
    INPUT_VEC_TYPE inv2 = *(INPUT_VEC_TYPE*)(input + input_idx + r + VEC_SIZE);
    INPUT_VEC_TYPE cosv1 = *(INPUT_VEC_TYPE*)(cos + cos_idx + r);
    INPUT_VEC_TYPE sinv1 = *(INPUT_VEC_TYPE*)(sin + sin_idx + r);
    INPUT_VEC_TYPE cosv2 = *(INPUT_VEC_TYPE*)(cos + cos_idx + r + VEC_SIZE);
    INPUT_VEC_TYPE sinv2 = *(INPUT_VEC_TYPE*)(sin + sin_idx + r + VEC_SIZE);

    float8 in1, in2, cos1, sin1, cos2, sin2;
    UNPACK_FLOAT_VEC_1(in1, inv1, inv2);
    UNPACK_FLOAT_VEC_2(in2, inv1, inv2);
    UNPACK_FLOAT_VEC_1(cos1, cosv1, cosv2);
    UNPACK_FLOAT_VEC_2(cos2, cosv1, cosv2);
    UNPACK_FLOAT_VEC_1(sin1, sinv1, sinv2);
    UNPACK_FLOAT_VEC_2(sin2, sinv1, sinv2);

    float8 out1 = cos1 * in1 - sin1 * in2;
    float8 out2 = sin2 * in1 + cos2 * in2;

    *(float8*)(output + output_idx + r) =
        (float8)(out1.s0, out2.s0, out1.s1, out2.s1, out1.s2, out2.s2, out1.s3, out2.s3);
    *(float8*)(output + output_idx + r + VEC_SIZE) =
        (float8)(out1.s4, out2.s4, out1.s5, out2.s5, out1.s6, out2.s6, out1.s7, out2.s7);
#elif VEC_SIZE == 16
    INPUT_VEC_TYPE inv1 = *(INPUT_VEC_TYPE*)(input + input_idx + r);
    INPUT_VEC_TYPE inv2 = *(INPUT_VEC_TYPE*)(input + input_idx + r + VEC_SIZE);
    INPUT_VEC_TYPE cosv1 = *(INPUT_VEC_TYPE*)(cos + cos_idx + r);
    INPUT_VEC_TYPE sinv1 = *(INPUT_VEC_TYPE*)(sin + sin_idx + r);
    INPUT_VEC_TYPE cosv2 = *(INPUT_VEC_TYPE*)(cos + cos_idx + r + VEC_SIZE);
    INPUT_VEC_TYPE sinv2 = *(INPUT_VEC_TYPE*)(sin + sin_idx + r + VEC_SIZE);

    INPUT_VEC_TYPE in1, in2, cos1, sin1, cos2, sin2;
    UNPACK_HALF16_VEC_1(in1, inv1, inv2);
    UNPACK_HALF16_VEC_2(in2, inv1, inv2);
    UNPACK_HALF16_VEC_1(cos1, cosv1, cosv2);
    UNPACK_HALF16_VEC_2(cos2, cosv1, cosv2);
    UNPACK_HALF16_VEC_1(sin1, sinv1, sinv2);
    UNPACK_HALF16_VEC_2(sin2, sinv1, sinv2);

    half16 out1 = cos1 * in1 - sin1 * in2;
    half16 out2 = sin2 * in1 + cos2 * in2;

    half16 outputv1, outputv2;
    PACK_HALF16_VEC_1(outputv1, out1, out2);
    PACK_HALF16_VEC_2(outputv2, out1, out2);

    *(half16*)(output + output_idx + r) = outputv1;
    *(half16*)(output + output_idx + r + VEC_SIZE) = outputv2;
#endif
}
#endif
