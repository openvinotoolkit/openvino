// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/f8_utils.cl"
#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS != 4 && OUTPUT_DIMS != 2
#error "dynamic_quantize_gpu_opt.cl: Unsupported output dimension"
#endif

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define CONVERT_UCHAR_N CAT(convert_uchar, VEC_SIZE)
#define CONVERT_CHAR_N CAT(convert_char, VEC_SIZE)
#define TO_TYPE_N_SAT_(type, n, x) _convert_##type##n##_sat(x)
#define TO_TYPE_N_SAT(type, n, x) TO_TYPE_N_SAT_(type, n, x)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT_TYPE_N(x) AS_TYPE_N(INPUT0_TYPE, VEC_SIZE, x)

#define IS_F8 (F8E5M2_OUTPUT || F8E4M3_OUTPUT)

// ***********************************************
#if DYNAMIC_QUANTIZAION_IMPL_MODE == MODE_SMALL_GS
// ***********************************************

#if ASYMMETRIC_QUANTIZATION
#error "UNIMPLMENTED: asymmetric quantization when group size is small"
#endif

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale
    ) {

#if OUTPUT_DIMS == 2
    const uint b = get_global_id(0);
    const uint f_grp = get_global_id(1);
    const uint input_offset = INPUT0_GET_INDEX(b, f_grp * QUANTIZE_GROUP_SIZE, 0, 0);
    const uint output_offset = OUTPUT_GET_INDEX(b, f_grp * QUANTIZE_GROUP_SIZE, 0, 0);
#else
    const uint bf = get_global_id(0);
    const uint b = bf / INPUT0_FEATURE_NUM;
    const uint f = bf % INPUT0_FEATURE_NUM;
    const uint y_grp = get_global_id(1);
    const uint input_offset = INPUT0_GET_INDEX(b, f, y_grp * QUANTIZE_GROUP_SIZE, 0);
    const uint output_offset = OUTPUT_GET_INDEX(b, f, y_grp * QUANTIZE_GROUP_SIZE, 0);

#endif
    const uint quantize_block = QUANTIZE_GROUP_SIZE / 4;
    half4 input_0[quantize_block];
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) quantized_value[quantize_block];
    half  max[quantize_block];

    unroll_for (uint i = 0 ; i < quantize_block; ++i) {
        input_0[i] = vload4(0, &input[input_offset + i * 4]);
        max[i] = fmax(fmax(fabs(input_0[i][0]), fabs(input_0[i][1])), fmax(fabs(input_0[i][2]), fabs(input_0[i][3])));
    }

    half max_value = fmax(0.001h, max[0]);
    for (uint i = 1; i < quantize_block; i++) {
        max_value = fmax(max_value, max[i]);
    }

#if IS_MXFP
    float out_dt_max_val_rounded_down = _convert_float(TO_OUTPUT1_TYPE(_convert_float(OUTPUT_VAL_MAX)));
    float max_val_rounded_down = _convert_float(TO_OUTPUT1_TYPE(max_value));
    half quan_scale = out_dt_max_val_rounded_down / max_val_rounded_down;
#else
    half quan_scale = _convert_half(OUTPUT_VAL_MAX) / max_value;
#endif // MXFP

    unroll_for (uint i = 0 ; i < quantize_block; ++i) {
#if IS_F8
        quantized_value[i] = TO_TYPE_N_SAT(OUTPUT_TYPE, 4, input_0[i] * (half4)quan_scale);
        // BLOCK_WRITEN(OUTPUT_TYPE, 4, output, output_offset + i * 4, quantized_value[i]);
        output[output_offset + i * 4] = AS_OUTPUT_TYPE(quantized_value[i].data[0]);
        output[output_offset + i * 4 + 1] = AS_OUTPUT_TYPE(quantized_value[i].data[1]);
        output[output_offset + i * 4 + 2] = AS_OUTPUT_TYPE(quantized_value[i].data[2]);
        output[output_offset + i * 4 + 3] = AS_OUTPUT_TYPE(quantized_value[i].data[3]);
#else
        quantized_value[i] = convert_char4(input_0[i] * (half4)quan_scale);
        vstore4(quantized_value[i], 0, &output[output_offset + i * 4]);
#endif // IS_F8
    }

#if OUTPUT_DIMS == 2
    output_scale[OUTPUT1_GET_INDEX(b, f_grp, 0, 0)] = TO_OUTPUT1_TYPE(1.0h / quan_scale);
#else
    output_scale[OUTPUT1_GET_INDEX(b, f, y_grp, 0)] = TO_OUTPUT1_TYPE(1.0h / quan_scale);
#endif
}

// ***********************************************
#elif DYNAMIC_QUANTIZAION_IMPL_MODE == MODE_LARGE_GS
// ***********************************************

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale
#if ASYMMETRIC_QUANTIZATION
    , __global OUTPUT2_TYPE* output_zp
#endif
    )
{
    const uint b = (uint)get_global_id(2);
    const uint f_grp = get_group_id(1);
    const uint sglid = get_sub_group_local_id();
    const uint local_id = (uint)get_local_id(1);
#if OUTPUT_DIMS == 2
    const uint input_offset = INPUT0_GET_INDEX (b, f_grp * QUANTIZE_GROUP_SIZE + VEC_SIZE * sglid, 0, 0);
    const uint output_offset = OUTPUT_GET_INDEX(b, f_grp * QUANTIZE_GROUP_SIZE + VEC_SIZE * sglid, 0, 0);
#else
    const uint input_offset = INPUT0_GET_INDEX (0, b, f_grp * QUANTIZE_GROUP_SIZE + VEC_SIZE * sglid, 0);
    const uint output_offset = OUTPUT_GET_INDEX(0, b, f_grp * QUANTIZE_GROUP_SIZE + VEC_SIZE * sglid, 0);
#endif

    const uint block_size = SIMD * VEC_SIZE;
#if OUTPUT_DIMS == 2
    const uint b_offset = b * INPUT0_BATCH_PITCH;
#else
    const uint b_offset = b * INPUT0_FEATURE_PITCH;
#endif
    const uint offset = b_offset + VEC_SIZE * sglid;

    __local half local_mem_max[QUANTIZE_GROUP_SIZE / block_size];
    __local half local_mem_min[QUANTIZE_GROUP_SIZE / block_size];

    MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) val;
    MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) abs_val;
    half grp_max = 0.001h;
    half grp_min = 0.001h;
    half max_value = 0.0h;
    half min_value = 0.0h;

    val = AS_INPUT_TYPE_N(VLOAD_N(0, input + input_offset + (local_id * block_size)));

#if ASYMMETRIC_QUANTIZATION
    unroll_for (int j = 0; j < VEC_SIZE; j++) {
        max_value = fmax(max_value, val[j]);
        min_value = fmin(min_value, val[j]);
    }
    grp_max = fmax(grp_max, max_value);
    grp_min = fmin(grp_min, min_value);
#else
    abs_val = fabs(val);

    unroll_for (int j = 0; j < VEC_SIZE; j++)
        max_value = fmax(max_value, abs_val[j]);

    grp_max = fmax(grp_max, max_value);
#endif

    max_value = sub_group_reduce_max(grp_max);
#if ASYMMETRIC_QUANTIZATION
    min_value = sub_group_reduce_min(grp_min);
#endif

    if (sglid == 0) {
        local_mem_max[local_id] = max_value;
#if ASYMMETRIC_QUANTIZATION
        local_mem_min[local_id] = min_value;
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = 0; j < QUANTIZE_GROUP_SIZE / block_size; j++) {
        max_value = fmax(max_value, local_mem_max[j]);
#if ASYMMETRIC_QUANTIZATION
        min_value = fmin(min_value, local_mem_min[j]);
#endif
    }

#if ASYMMETRIC_QUANTIZATION
    OUTPUT1_TYPE scale = (OUTPUT1_TYPE)((CHAR_MAX - CHAR_MIN) / (max_value - min_value));
    OUTPUT2_TYPE zp = (OUTPUT2_TYPE)(-min_value * scale);
#else
    OUTPUT1_TYPE scale = _convert_half(OUTPUT_VAL_MAX) / max_value;
#endif

    val *= scale;
#if ASYMMETRIC_QUANTIZATION
    val += zp;
    VSTORE_N(CAT(CONVERT_UCHAR_N, _rte)(val), 0, output + output_offset + (local_id * block_size));
#else // ASYMMETRIC_QUANTIZATION
#if IS_F8
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE) out = TO_TYPE_N_SAT(OUTPUT_TYPE, VEC_SIZE, val);
    // BLOCK_WRITEN(OUTPUT_TYPE, VEC_SIZE, output + output_offset + (local_id * block_size), 0, out);
    for (uint i = 0; i < VEC_SIZE; ++i)
        output[output_offset + (local_id * block_size) + i] = AS_OUTPUT_TYPE(out.data[j]);
#else // IS_F8
    VSTORE_N(CAT(CONVERT_CHAR_N, _rte)(val), 0, output + output_offset + (local_id * block_size));
#endif // IS_F8
#endif // ASYMMETRIC_QUANTIZATION

    if (sglid == 0 && local_id == 0) {
#if OUTPUT_DIMS == 2
        const int output_idx = OUTPUT1_GET_INDEX(b, f_grp, 0, 0);
#else
        const int output_idx = OUTPUT1_GET_INDEX(0, b, f_grp, 0);
#endif

        output_scale[output_idx] = 1.0h / scale;
#if ASYMMETRIC_QUANTIZATION
        output_zp[output_idx] = convert_uchar_rte(zp);
#endif
    }
}

// ***********************************************
#elif DYNAMIC_QUANTIZAION_IMPL_MODE == MODE_PER_TOKEN
// ***********************************************

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale
#if ASYMMETRIC_QUANTIZATION
    , __global OUTPUT2_TYPE* output_zp
#endif
    )
{
    const uint bf = (uint)get_global_id(2);
    const uint sglid = get_sub_group_local_id();
    const uint local_id = (uint)get_local_id(1);

    const uint block_size = SIMD * VEC_SIZE;
#if OUTPUT_DIMS == 2
    const uint b_offset = bf * INPUT0_BATCH_PITCH;
#else
    const uint b_offset = bf * INPUT0_FEATURE_PITCH;
#endif
    const uint offset = b_offset + VEC_SIZE * sglid;

    const uint iteration = ALIGNED_BLOCK_NUM / BLOCK_NUM;

    __local half local_mem_max[BLOCK_NUM];
    __local half local_mem_min[BLOCK_NUM];

    MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) val[iteration];
    MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) abs_val;
    half grp_max = 0.001h;
    half grp_min = 0.001h;
    half max_value = 0.0h;
    half min_value = 0.0h;

    unroll_for(int i = 0; i < iteration; ++i) {
        if ((local_id * iteration + i) >= TOTAL_BLOCK_NUM)
            continue;

        val[i] = AS_INPUT_TYPE_N(VLOAD_N(0, input + offset + ((local_id * iteration + i) * block_size)));
#if ASYMMETRIC_QUANTIZATION
        unroll_for (int j = 0; j < VEC_SIZE; j++) {
            max_value = fmax(max_value, val[i][j]);
            min_value = fmin(min_value, val[i][j]);
        }
        grp_max = fmax(grp_max, max_value);
        grp_min = fmin(grp_min, min_value);
#else
        abs_val = fabs(val[i]);

        unroll_for (int j = 0; j < VEC_SIZE; j++)
            max_value = fmax(max_value, abs_val[j]);

        grp_max = fmax(grp_max, max_value);
#endif
    }

    max_value = sub_group_reduce_max(grp_max);
#if ASYMMETRIC_QUANTIZATION
    min_value = sub_group_reduce_min(grp_min);
#endif

    if (sglid == 0) {
        local_mem_max[local_id] = max_value;
#if ASYMMETRIC_QUANTIZATION
        local_mem_min[local_id] = min_value;
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = 0; j < BLOCK_NUM; j++) {
        max_value = fmax(max_value, local_mem_max[j]);
#if ASYMMETRIC_QUANTIZATION
        min_value = fmin(min_value, local_mem_min[j]);
#endif
    }

#if ASYMMETRIC_QUANTIZATION
    OUTPUT1_TYPE scale = (OUTPUT1_TYPE)((CHAR_MAX - CHAR_MIN) / (max_value - min_value));
    OUTPUT2_TYPE zp = (OUTPUT2_TYPE)(-min_value * scale);
#else
    OUTPUT1_TYPE scale = _convert_half(OUTPUT_VAL_MAX) / max_value;
#endif


    unroll_for(int i = 0; i < iteration; ++i) {
        if ((local_id * iteration + i) >= TOTAL_BLOCK_NUM)
            continue;

        val[i] *= scale;
#if ASYMMETRIC_QUANTIZATION
        val[i] += zp;
        VSTORE_N(CAT(CONVERT_UCHAR_N, _rte)(val[i]), 0, output + offset + ((local_id * iteration + i) * block_size));
#else // ASYMMETRIC_QUANTIZATION
#if IS_F8
        MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE) out = TO_TYPE_N_SAT(OUTPUT_TYPE, VEC_SIZE, val[i]);
        // VSTORE_N(AS_TYPE_N(char, VEC_SIZE, out), 0, (char*)output + offset + ((local_id * iteration + i) * block_size));
        // BLOCK_WRITEN(OUTPUT_TYPE, VEC_SIZE, output + offset + ((local_id * iteration + i) * block_size), 0, out);
        for (uint j = 0; j < VEC_SIZE; ++j)
            output[offset + ((local_id * iteration + i) * block_size) + j] = AS_OUTPUT_TYPE(out.data[j]);
#else // IS_F8
        VSTORE_N(CAT(CONVERT_CHAR_N, _rte)(val[i]), 0, output + offset + ((local_id * iteration + i) * block_size));
#endif // IS_F8
#endif // ASYMMETRIC_QUANTIZATION
    }

    if (sglid == 0 && local_id == 0) {
        output_scale[bf] = 1.0h / scale;
#if ASYMMETRIC_QUANTIZATION
        output_zp[bf] = convert_uchar_rte(zp);
#endif
    }
}

#else   // DYNAMIC_QUANTIZAION_IMPL_MODE
#error Unimplemented IMPL_MODE
#endif  // DYNAMIC_QUANTIZAION_IMPL_MODE
