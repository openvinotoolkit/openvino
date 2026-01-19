// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS != 4 && OUTPUT_DIMS != 2
#error "dynamic_quantize_gpu_opt.cl: Unsupported output dimension"
#endif

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define CONVERT_UCHAR_N CAT(convert_uchar, VEC_SIZE)
#define CONVERT_CHAR_N CAT(convert_char, VEC_SIZE)
#define CONVERT_INT_N CAT(convert_int, VEC_SIZE)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT_TYPE_N(x) AS_TYPE_N(INPUT0_TYPE, VEC_SIZE, x)
#define ACT_MIN_VAL 0.003h      // Too small value may generate inf during 127/ACT_MIN_VAL

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
    char4 quantized_value[quantize_block];
    half  max[quantize_block];

    unroll_for (uint i = 0 ; i < quantize_block; ++i) {
        input_0[i] = vload4(0, &input[input_offset + i * 4]);
        max[i] = fmax(fmax(fabs(input_0[i][0]), fabs(input_0[i][1])), fmax(fabs(input_0[i][2]), fabs(input_0[i][3])));
    }

    half max_value = fmax(ACT_MIN_VAL, max[0]);
    for (uint i = 1; i < quantize_block; i++) {
        max_value = fmax(max_value, max[i]);
    }

    half quan_scale = 127.0h / max_value;
    FOR_PRECOMPUTED_REDUCTION(int precomputed_reduction = 0);

    unroll_for (uint i = 0 ; i < quantize_block; ++i) {
        quantized_value[i] = convert_char4_rte(input_0[i] * (half4)quan_scale);
        FOR_PRECOMPUTED_REDUCTION(precomputed_reduction += quantized_value[i][0] + quantized_value[i][1] + quantized_value[i][2] + quantized_value[i][3]);
        vstore4(quantized_value[i], 0, &output[output_offset + i * 4]);
    }

#if OUTPUT_DIMS == 2
    const uint output_idx = OUTPUT1_GET_INDEX(b, f_grp, 0, 0);
#else
    const uint output_idx = OUTPUT1_GET_INDEX(b, f, y_grp, 0);
#endif
    output_scale[output_idx] = 1.0h / quan_scale;

    FOR_PRECOMPUTED_REDUCTION(output_precomputed_reduction[output_idx] = precomputed_reduction);
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
    OUTPUT1_TYPE scale = 127.0h / max_value;
#endif

    val *= scale;
#if ASYMMETRIC_QUANTIZATION
    val += zp;
    VSTORE_N(CAT(CONVERT_UCHAR_N, _rte)(val), 0, output + output_offset + (blockid * block_size));
#else
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
        FOR_PRECOMPUTED_REDUCTION(output_precomputed_reduction[output_idx] = precomputed_reduction);
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
    OUTPUT1_TYPE scale = 127.0h / max_value;
#endif


    unroll_for(int i = 0; i < iteration; ++i) {
        if ((local_id * iteration + i) >= TOTAL_BLOCK_NUM)
            continue;

        val[i] *= scale;
#if ASYMMETRIC_QUANTIZATION
        val[i] += zp;
        VSTORE_N(CAT(CONVERT_UCHAR_N, _rte)(val[i]), 0, output + offset + ((local_id * iteration + i) * block_size));
#else
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
