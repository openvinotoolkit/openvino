// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/int4_utils.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"


#define UINT4_RANGE 15

#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_kv_cache.cl: Unsupported output dimension"
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

    // The innermost dimension is always processed in the loop inside the kernel
    const uint x = 0;

    half grp_max = 0.001h;
    half max_value = INPUT0_VAL_MIN;
    half min_value = INPUT0_VAL_MAX;

    half val[INNERMOST_DIM_VALUE / SUBGROUP_SIZE];

    const uint input_offset = INPUT0_GET_INDEX(b, f, y, x);
    unroll_for (uint i = 0; i < INNERMOST_DIM_VALUE / SUBGROUP_SIZE; i++) {
        val[i] = INPUT_BLOCK_READ(input, input_offset + i * SUBGROUP_SIZE);
#if ASYMMETRIC_QUANTIZATION || IS_INT4_COMPRESSED
        max_value = fmax(max_value, val[i]);
        min_value = fmin(min_value, val[i]);
#else
        max_value = fmax(max_value, fabs(val[i]));
#endif
    }
#if !ASYMMETRIC_QUANTIZATION && !IS_INT4_COMPRESSED
    max_value = fmax(max_value, grp_max);
#endif

#ifdef APPEND_MODE
    APPEND_AXIS_NAME += axis_offset;
#endif

#if IS_INT4_COMPRESSED
    // 4-bit unsigned asymmetric quantization: map [min, max] to [0, 15].
    // Two INT4 values are packed per byte using SUBGROUP_SIZE-stride grouping:
    //   output byte at physical offset (k * SUBGROUP_SIZE + sglid) holds:
    //     lo nibble = quantized val[(2k)   * SUBGROUP_SIZE + sglid]
    //     hi nibble = quantized val[(2k+1) * SUBGROUP_SIZE + sglid]
    min_value = work_group_reduce_min(min_value);
    max_value = work_group_reduce_max(max_value);

    ACCUMULATOR_TYPE diff_value = max_value == min_value ? (ACCUMULATOR_TYPE)(grp_max)
                                                         : (ACCUMULATOR_TYPE)(max_value - min_value);
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((UINT4_RANGE) / diff_value);
    ACCUMULATOR_TYPE zp_tmp    = (ACCUMULATOR_TYPE)(-min_value * scale_tmp); // maps min -> 0, max -> UINT4_RANGE

    // INT4 packed buffer: the output layout uses i8 with full head_size shape.
    // Use element-level offset directly (same stride as layout) so that SDPA
    // can address rows with the standard GET_INDEX pitch.
    const uint output_offset = OUTPUT_GET_INDEX(b, f, y, x);
    // Pairs of consecutive SUBGROUP_SIZE blocks are packed together.
    // Process complete pairs first (handles HEAD_SIZE that is multiple of 2*SUBGROUP_SIZE).
#define NUM_SUBGROUP_CHUNKS (INNERMOST_DIM_VALUE / SUBGROUP_SIZE)
#define NUM_FULL_PAIRS (NUM_SUBGROUP_CHUNKS / 2)
    unroll_for (uint i = 0; i < NUM_FULL_PAIRS * 2; i += 2) {
        uchar q0 = (uchar)clamp(convert_int_rte((float)val[i]     * scale_tmp + zp_tmp), 0, UINT4_RANGE);
        uchar q1 = (uchar)clamp(convert_int_rte((float)val[i + 1] * scale_tmp + zp_tmp), 0, UINT4_RANGE);
        // Pack: lo nibble = q0, hi nibble = q1
        char packed = cvt_uint8x2_to_uint4x2((uchar2)(q0, q1));
        OUTPUT_BLOCK_WRITE(output, output_offset + (i / 2) * SUBGROUP_SIZE, packed);
    }
#if (NUM_SUBGROUP_CHUNKS % 2) != 0
    // Handle the last odd chunk: pack low nibble only, zero-pad high nibble.
    {
        const uint i = NUM_FULL_PAIRS * 2;
        uchar q0 = (uchar)clamp(convert_int_rte((float)val[i] * scale_tmp + zp_tmp), 0, UINT4_RANGE);
        char packed = cvt_uint8x2_to_uint4x2((uchar2)(q0, 0));
        OUTPUT_BLOCK_WRITE(output, output_offset + (i / 2) * SUBGROUP_SIZE, packed);
    }
#endif
#undef NUM_SUBGROUP_CHUNKS
#undef NUM_FULL_PAIRS

    const uint scale_idx = FUNC_CALL(get_scales_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, y, x);
    if (grouped_indexes == 0 && sglid == 0) {
        output_scale[scale_idx]     = (OUTPUT1_TYPE)(1.0f / scale_tmp); // dequant scale
        output_scale[scale_idx + 1] = (OUTPUT1_TYPE)(zp_tmp);           // zero-point
    }

#else  // !IS_INT4_COMPRESSED — original INT8 path

#if ASYMMETRIC_QUANTIZATION
    min_value = work_group_reduce_min(min_value);
    max_value = work_group_reduce_max(max_value);

    // If the range of input data is zero, it is adjusted to the minimum value(0.001).
    ACCUMULATOR_TYPE diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
    OUTPUT1_TYPE scale = (OUTPUT1_TYPE)(scale_tmp);
    OUTPUT1_TYPE zp = (OUTPUT1_TYPE)(zp_tmp);

#else
    max_value = work_group_reduce_max(max_value);
    OUTPUT1_TYPE scale = 127.0h / max_value;
#endif

    const uint output_offset = OUTPUT_GET_INDEX(b, f, y, x);
    unroll_for (uint i = 0; i < INNERMOST_DIM_VALUE / SUBGROUP_SIZE; i++) {
#if ASYMMETRIC_QUANTIZATION
        OUTPUT_TYPE res = convert_char_rte(val[i] * scale + zp);
#else
        OUTPUT_TYPE res = convert_char_rte(val[i] * scale);
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

    #if OUTPUT2_IS_FP
        output_zp[scale_idx] = zp;
    #else
        output_zp[scale_idx] = convert_char_rte(zp);
    #endif

#endif
#else
        output_scale[scale_idx] = 1.0h / scale;
#endif
    }

#endif  // IS_INT4_COMPRESSED
}
