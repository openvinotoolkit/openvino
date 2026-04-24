// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define IS_F8 (F8E5M2_OUTPUT || F8E4M3_OUTPUT)

#include "include/batch_headers/fetch_data.cl"
#if IS_F8
#include "include/batch_headers/f8_utils.cl"
#endif

#if OUTPUT_DIMS != 4 && OUTPUT_DIMS != 2
#error "dynamic_quantize_gpu_opt.cl: Unsupported output dimension"
#endif

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define CONVERT_UCHAR_N CAT(convert_uchar, VEC_SIZE)
#define CONVERT_CHAR_N CAT(convert_char, VEC_SIZE)
#define CONVERT_INT_N CAT(convert_int, VEC_SIZE)
#define TO_TYPE_N_(type, n, x) convert_##type##n(x)
#define TO_TYPE_N(type, n, x) TO_TYPE_N_(type, n, x)
#define TO_TYPE_N_SAT_(type, n, x) convert_##type##n##_sat(x)
#define TO_TYPE_N_SAT(type, n, x) TO_TYPE_N_SAT_(type, n, x)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT_TYPE_N(x) AS_TYPE_N(INPUT0_TYPE, VEC_SIZE, x)
#if IS_F8
    #define SCALE_TYPE float
    #define TO_SCALE_TYPE(x) _convert_float(x)
    #define ACT_MIN_VAL 0.000000059604645h // min half dtype val
#else
    #define SCALE_TYPE half
    #define TO_SCALE_TYPE(x) convert_half(x)
    #define ACT_MIN_VAL 0.003h      // Too small value may generate inf during 127/ACT_MIN_VAL
#endif

#if GENERATE_PRECOMPUTED_REDUCTION
    #define FOR_PRECOMPUTED_REDUCTION(x)  x
#else
    #define FOR_PRECOMPUTED_REDUCTION(x)
#endif


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
#if GENERATE_PRECOMPUTED_REDUCTION
    , __global OUTPUT2_TYPE* output_precomputed_reduction
#endif
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

    half max_value = fmax(ACT_MIN_VAL, max[0]);
    for (uint i = 1; i < quantize_block; i++) {
        max_value = fmax(max_value, max[i]);
    }

#if IS_MXFP
    SCALE_TYPE quan_scale = (SCALE_TYPE)(exp2(floor(log2(_convert_float(OUTPUT_VAL_MAX) / _convert_float(max_value)))));
#else
    SCALE_TYPE quan_scale = TO_SCALE_TYPE(OUTPUT_VAL_MAX) / max_value;
    FOR_PRECOMPUTED_REDUCTION(int precomputed_reduction = 0);
#endif // MXFP

    unroll_for (uint i = 0 ; i < quantize_block; ++i) {
#if IS_F8
        quantized_value[i] = TO_TYPE_N_SAT(OUTPUT_TYPE, 4, convert_float4(input_0[i]) * (MAKE_VECTOR_TYPE(SCALE_TYPE, 4))quan_scale);
        vstore4(quantized_value[i].data, 0, (char*)(&output[output_offset + i * 4]));
#else
        quantized_value[i] = convert_char4_rte(input_0[i] * (half4)quan_scale);
        FOR_PRECOMPUTED_REDUCTION(precomputed_reduction += quantized_value[i][0] + quantized_value[i][1] + quantized_value[i][2] + quantized_value[i][3]);
        vstore4(quantized_value[i], 0, &output[output_offset + i * 4]);
#endif // IS_F8
    }

#if OUTPUT_DIMS == 2
    const uint output_idx = OUTPUT1_GET_INDEX(b, f_grp, 0, 0);
#else
    const uint output_idx = OUTPUT1_GET_INDEX(b, f, y_grp, 0);
#endif
    output_scale[output_idx] = TO_OUTPUT1_TYPE(1.0h / quan_scale);

#if !(IS_MXFP)
    FOR_PRECOMPUTED_REDUCTION(output_precomputed_reduction[output_idx] = precomputed_reduction);
#endif
}

// ***********************************************
#elif DYNAMIC_QUANTIZAION_IMPL_MODE == MODE_LARGE_GS
// ***********************************************

#if ASYMMETRIC_QUANTIZATION != 0 && GENERATE_PRECOMPUTED_REDUCTION != 0
#error "UNIMPLMENTED: asymmetric quantization with precomputed_reduction generation"
#endif

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(dynamic_quantize_gpu_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale
#if ASYMMETRIC_QUANTIZATION
    , __global OUTPUT2_TYPE* output_zp
#endif
#if GENERATE_PRECOMPUTED_REDUCTION
    , __global OUTPUT2_TYPE* output_precomputed_reduction
#endif
    )
{
    const uint b = (uint)get_global_id(2);
    const uint f_grp = get_global_id(1) * VEC_SIZE * SIMD / QUANTIZE_GROUP_SIZE;
    const uint sglid = get_sub_group_local_id();
    const uint blockid = (uint)get_global_id(1) % (QUANTIZE_GROUP_SIZE / VEC_SIZE / SIMD);
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

    const uint local_id = get_local_id(1);
    __local half local_mem_max[BLOCK_NUM];
    __local half local_mem_min[BLOCK_NUM];

    MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) val;
    MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) abs_val;
    half grp_max = ACT_MIN_VAL;
    half grp_min = ACT_MIN_VAL;
    half max_value = 0.0h;
    half min_value = 0.0h;
    val = AS_INPUT_TYPE_N(VLOAD_N(0, input + input_offset + (blockid * block_size)));

#if ASYMMETRIC_QUANTIZATION
    unroll_for (int j = 0; j < VEC_SIZE; j++) {
        max_value = fmax(max_value, val[j]);
        min_value = fmin(min_value, val[j]);
    }
    grp_max = fmax(grp_max, max_value);
    grp_min = fmin(grp_min, min_value);
#else
    abs_val = fabs(val);

    unroll_for (int j = 0; j < VEC_SIZE; j++) {
        max_value = fmax(max_value, abs_val[j]);
    }

    grp_max = fmax(grp_max, max_value);
#endif

    max_value = sub_group_reduce_max(grp_max);
#if ASYMMETRIC_QUANTIZATION
    min_value = sub_group_reduce_min(grp_min);
#endif

    const uint block_offset_idx = local_id * QUANTIZE_GROUP_SIZE / block_size;
    if (sglid == 0) {
        local_mem_max[block_offset_idx + blockid] = max_value;
#if ASYMMETRIC_QUANTIZATION
        local_mem_min[block_offset_idx + blockid] = min_value;
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = 0; j < QUANTIZE_GROUP_SIZE / block_size; j++) {
        max_value = fmax(max_value, local_mem_max[block_offset_idx + j]);
#if ASYMMETRIC_QUANTIZATION
        min_value = fmin(min_value, local_mem_min[block_offset_idx + j]);
#endif
    }

#if ASYMMETRIC_QUANTIZATION
    OUTPUT1_TYPE scale = (OUTPUT1_TYPE)((CHAR_MAX - CHAR_MIN) / (max_value - min_value));
    OUTPUT2_TYPE zp = (OUTPUT2_TYPE)(-min_value * scale);
#else
    SCALE_TYPE scale = TO_SCALE_TYPE(OUTPUT_VAL_MAX) / max_value;
#endif

#if IS_F8
    val = TO_TYPE_N(INPUT0_TYPE, VEC_SIZE, TO_TYPE_N(SCALE_TYPE, VEC_SIZE, val) * (MAKE_VECTOR_TYPE(SCALE_TYPE, VEC_SIZE))scale);
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE) out = TO_TYPE_N_SAT(OUTPUT_TYPE, VEC_SIZE, val);
    VSTORE_N(out.data, 0, (char*)(&output[output_offset + (blockid * block_size)]));
#elif ASYMMETRIC_QUANTIZATION
    val *= scale;
    val += zp;
    VSTORE_N(CAT(CONVERT_UCHAR_N, _rte)(val), 0, output + output_offset + (blockid * block_size));
#else // i8 symmetric
    val *= scale;
    VSTORE_N(CAT(CONVERT_CHAR_N, _rte)(val), 0, output + output_offset + (blockid * block_size));
#endif

#if GENERATE_PRECOMPUTED_REDUCTION
    // TODO: Optimize this part
    int precomputed_reduction = 0;
    MAKE_VECTOR_TYPE(OUTPUT2_TYPE, VEC_SIZE) val_int = CAT(CONVERT_INT_N, _rte)(val);
    unroll_for (int j = 0; j < VEC_SIZE; j++) {
        precomputed_reduction += val_int[j];
    }
    precomputed_reduction = sub_group_reduce_add(precomputed_reduction);
#endif

    if (sglid == 0 && blockid == 0) {
#if OUTPUT_DIMS == 2
        const int output_idx = OUTPUT1_GET_INDEX(b, f_grp, 0, 0);
#else
        const int output_idx = OUTPUT1_GET_INDEX(0, b, f_grp, 0);
#endif

        output_scale[output_idx] = 1.0h / scale;
#if ASYMMETRIC_QUANTIZATION
        output_zp[output_idx] = convert_uchar_rte(zp);
#endif
#if !(IS_MXFP)
        FOR_PRECOMPUTED_REDUCTION(output_precomputed_reduction[output_idx] = precomputed_reduction);
#endif
    }
}

// ***********************************************
#elif DYNAMIC_QUANTIZAION_IMPL_MODE == MODE_PER_TOKEN
// ***********************************************

#if GENERATE_PRECOMPUTED_REDUCTION != 0
#error "UNIMPLMENTED: precomputed_reduction generation"
#endif

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
    half grp_max = ACT_MIN_VAL;
    half grp_min = ACT_MIN_VAL;
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
    SCALE_TYPE scale = TO_SCALE_TYPE(OUTPUT_VAL_MAX) / max_value;
#endif


    unroll_for(int i = 0; i < iteration; ++i) {
        if ((local_id * iteration + i) >= TOTAL_BLOCK_NUM)
            continue;

        val[i] = TO_TYPE_N(INPUT0_TYPE, VEC_SIZE, TO_TYPE_N(SCALE_TYPE, VEC_SIZE, val[i]) * (MAKE_VECTOR_TYPE(SCALE_TYPE, VEC_SIZE))scale);
#if IS_F8
        MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE) out = TO_TYPE_N_SAT(OUTPUT_TYPE, VEC_SIZE, val[i]);
        VSTORE_N(out.data, 0, (char*)(&output[offset + ((local_id * iteration + i) * block_size)]));
#elif ASYMMETRIC_QUANTIZATION
        val[i] += zp;
        VSTORE_N(CAT(CONVERT_UCHAR_N, _rte)(val[i]), 0, output + offset + ((local_id * iteration + i) * block_size));
#else // i8 symmetric
        VSTORE_N(CAT(CONVERT_CHAR_N, _rte)(val[i]), 0, output + offset + ((local_id * iteration + i) * block_size));
#endif
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
