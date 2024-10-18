// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"


#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_opt.cl: Unsupported output dimension"
#endif

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define CONVERT_CHAR_N CAT(convert_char, VEC_SIZE)
#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT_TYPE_N(x) AS_TYPE_N(INPUT0_TYPE, VEC_SIZE, x)


inline uint FUNC(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint y, uint x) {
    return OUTPUT1_GET_INDEX(b, f, y, x);
}

inline uint FUNC(get_scales_offset)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint y, uint x) {
#ifdef SCALES_OUTPUT_ORDER
    return FUNC_CALL(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_TENSOR SCALES_OUTPUT_ORDER);
#else
    return FUNC_CALL(get_scales_offset_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, y, x);
#endif
}

#define SUBGROUP_SIZE 16
#define INNERMOST_DIM_VALUE INPUT0_SIZE_X
#define INPUT_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)

__attribute__((reqd_work_group_size(SUBGROUP_SIZE, SUBGROUPS_NUMBER, 1)))
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(dynamic_quantize_gpu_kv_cache)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale
#if ASYMMETRIC_QUANTIZATION && !GROUP_SCALES_WITH_ZP
    , __global OUTPUT2_TYPE* output_zp
#endif
#ifdef APPEND_MODE
    , const uint axis_offset
#endif
    )
{
    const uint sglid = get_sub_group_local_id();
    const uint grouped_indexes = get_global_id(1);
    const uint batch_indexes = get_global_id(2);

    DECLARE_BATCHED_DIMS_INDEXES(batch_indexes);
    DECLARE_GROUPED_DIMS_INDEXES(grouped_indexes);

    // the innermost dimension is always handled in the loop inside the kernel
    const uint x = 0;

    half max_value = INPUT0_VAL_MIN;
    half min_value = INPUT0_VAL_MAX;

    half val[INNERMOST_DIM_VALUE / SUBGROUP_SIZE];

    const uint input_offset = INPUT0_GET_INDEX(b, f, y, x);
    unroll_for (uint i = 0; i < INNERMOST_DIM_VALUE / SUBGROUP_SIZE; i++) {
        val[i] = INPUT_BLOCK_READ(input, input_offset + i * SUBGROUP_SIZE);
#if ASYMMETRIC_QUANTIZATION
        max_value = fmax(max_value, val[i]);
        min_value = fmin(min_value, val[i]);
#else
        max_value = fmax(max_value, fabs(val[i]));
#endif
    }

#if ASYMMETRIC_QUANTIZATION
    min_value = work_group_reduce_min(min_value);
    max_value = work_group_reduce_max(max_value);
    OUTPUT1_TYPE scale = (OUTPUT1_TYPE)((CHAR_MAX - CHAR_MIN) / (max_value - min_value));
    OUTPUT1_TYPE zp = (OUTPUT1_TYPE)(-min_value * scale) - CHAR_MAX;
#else
    max_value = work_group_reduce_max(max_value);
    OUTPUT1_TYPE scale = 127.0h / max_value;
#endif

#ifdef APPEND_MODE
    APPEND_AXIS_NAME += axis_offset;
#endif

    const uint output_offset = OUTPUT_GET_INDEX(b, f, y, x);
    unroll_for (uint i = 0; i < INNERMOST_DIM_VALUE / SUBGROUP_SIZE; i++) {
#if ASYMMETRIC_QUANTIZATION
        OUTPUT_TYPE res = convert_char(val[i] * scale + zp);
#else
        OUTPUT_TYPE res = convert_char(val[i] * scale);
#endif
        OUTPUT_BLOCK_WRITE(output, output_offset + i * SUBGROUP_SIZE, res);
    }

    const uint scale_idx = FUNC_CALL(get_scales_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, y, x);

    if (grouped_indexes == 0 && sglid == 0) {
#if ASYMMETRIC_QUANTIZATION
        output_scale[scale_idx] = 1.0h / scale;
#if GROUP_SCALES_WITH_ZP
        output_scale[scale_idx + 1] = zp;
#else
        output_zp[scale_idx] = zp;
#endif
#else
        output_scale[scale_idx] = 1.0h / scale;
#endif
    }
}
