// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define UINT64_MAX 0xFFFFFFFFFFFFFFFF

#if ASYMMETRIC_QUANTIZATION && UNSIGNED_OUTPUT
    #define TO_OUTPUT_TYPE_RTE(val)  convert_uchar_rte(val)
    #define TO_OUTPUT_VEC_TYPE_RTE(val)  convert_uchar8_rte(val)
#else
    #define TO_OUTPUT_TYPE_RTE(val)  convert_char_rte(val)
    #define TO_OUTPUT_VEC_TYPE_RTE(val)  convert_char8_rte(val)
#endif

#if OUTPUT_DIMS != 4
#error "dynamic_quantize_gpu_ref.cl: Unsupported output dimension"
#endif

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

KERNEL(dynamic_quantize_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* output_scale
#if ASYMMETRIC_QUANTIZATION && !GROUP_SCALES_WITH_ZP
    , __global OUTPUT2_TYPE* output_zp
#endif
)
{
    const uint bf = (uint)get_global_id(0);
    const uint b = bf / INPUT0_FEATURE_NUM;
    const uint f = bf % INPUT0_FEATURE_NUM;
    const uint out_y = (uint)get_global_id(1);
    const uint y = out_y * GROUP_SIZE_DIM2;     // quantization may be grouped for y axis
    const uint x = (uint)get_global_id(2);
#ifdef SCALES_OUTPUT_ORDER
    const uint scale_idx = FUNC_CALL(get_scales_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, out_y, x);
#else
    const uint scale_idx = OUTPUT1_GET_INDEX_SAFE(b, f, out_y, x);
#endif

    half grp_max = 0.001h;
    half max_val = INPUT0_VAL_MIN;
    half min_val = INPUT0_VAL_MAX;
    for (int b_off = 0; b_off < (GROUP_SIZE_DIM0 == 1 ? 1 : INPUT0_BATCH_NUM); b_off++) {
    for (int f_off = 0; f_off < (GROUP_SIZE_DIM1 == 1 ? 1 : INPUT0_FEATURE_NUM); f_off++) {
    for (int y_off = 0; y_off < (GROUP_SIZE_DIM2 == UINT64_MAX ? INPUT0_SIZE_Y : GROUP_SIZE_DIM2); y_off++) {
        // It is assumed that grouped quantization happens only for 3d input case where we don't have x axis
#if GROUP_SIZE_DIM3 == 1
        const uint offset = INPUT0_GET_INDEX(b + b_off, f + f_off, y + y_off, x);
        half val = input[offset];
#if ASYMMETRIC_QUANTIZATION
        max_val = fmax(max_val, val);
        min_val = fmin(min_val, val);
#else
        half abs_val = fabs(val);
        max_val = fmax(max_val, abs_val);
#endif
#else
        const uint offset = INPUT0_GET_INDEX(b + b_off, f + f_off, y + y_off, 0);
        int x;
        for (x = 0; x < INPUT0_SIZE_X / 8; x++) {
            half8 val = as_half8(vload8(0, (ushort*)input + offset + x * 8));
            half8 abs_val = fabs(val);
            for (int j = 0; j < 8; j++) {
#if ASYMMETRIC_QUANTIZATION
                max_val = fmax(max_val, val[j]);
                min_val = fmin(min_val, val[j]);
#else
                max_val = fmax(max_val, abs_val[j]);
#endif
            }
        }
        x *= 8;
        for (; x < INPUT0_SIZE_X; x++) {
            half val = input[offset + x];
#if ASYMMETRIC_QUANTIZATION
            max_val = fmax(max_val, val);
            min_val = fmin(min_val, val);
#else
            max_val = fmax(max_val, fabs(val));
#endif
        }
#endif
    }
    }
    }
#if !ASYMMETRIC_QUANTIZATION
    max_val = fmax(max_val, grp_max);
#endif

#if ASYMMETRIC_QUANTIZATION
    // If the range of input data is zero, it is adjusted to the minimum value(0.001).
    ACCUMULATOR_TYPE diff_value = max_val == min_val ? (grp_max) : (max_val - min_val);
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
#   if UNSIGNED_OUTPUT
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_val * scale_tmp);
#   else // !UNSIGNED_OUTPUT
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_val * scale_tmp) + CHAR_MIN;
#   endif
    OUTPUT1_TYPE scale = (OUTPUT1_TYPE)(scale_tmp);
    OUTPUT1_TYPE zp = (OUTPUT1_TYPE)(zp_tmp);
#else  // !ASYMMETRIC_QUANTIZATION
    max_val = work_group_reduce_max(max_val);
    OUTPUT1_TYPE scale = 127.0h / max_val;
#endif

    for (int b_off = 0; b_off < (GROUP_SIZE_DIM0 == 1 ? 1 : INPUT0_BATCH_NUM); b_off++) {
    for (int f_off = 0; f_off < (GROUP_SIZE_DIM1 == 1 ? 1 : INPUT0_FEATURE_NUM); f_off++) {
    for (int y_off = 0; y_off < (GROUP_SIZE_DIM2 == UINT64_MAX ? INPUT0_SIZE_Y : GROUP_SIZE_DIM2); y_off++) {
#if GROUP_SIZE_DIM3 == 1
        const uint in_offset = INPUT0_GET_INDEX(b + b_off, f + f_off, y + y_off, x);
        const uint out_offset = OUTPUT_GET_INDEX(b + b_off, f + f_off, y + y_off, x);

        half val = input[in_offset];
        val *= scale;
#if ASYMMETRIC_QUANTIZATION
        val += zp;
#endif
        output[out_offset] = TO_OUTPUT_TYPE_RTE(val);
#else
        const uint in_offset = INPUT0_GET_INDEX(b + b_off, f + f_off, y + y_off, 0);
        const uint out_offset = OUTPUT_GET_INDEX(b + b_off, f + f_off, y + y_off, 0);
        int x;
        for (x = 0; x < INPUT0_SIZE_X / 8; x++) {
            half8 val = as_half8(vload8(0, (ushort*)input + in_offset + x * 8));
            val *= scale;
#if ASYMMETRIC_QUANTIZATION
            val += zp;
#endif
            vstore8(TO_OUTPUT_VEC_TYPE_RTE(val), 0, output + out_offset + x * 8);
        }
        x *= 8;
        for (; x < INPUT0_SIZE_X; x++) {
            half val = input[in_offset + x];
            val *= scale;
#if ASYMMETRIC_QUANTIZATION
            val += zp;
#endif
            output[out_offset + x] = TO_OUTPUT_TYPE_RTE(val);
        }
#endif
    }
    }
    }

    output_scale[scale_idx] = 1.0h / scale;
#if ASYMMETRIC_QUANTIZATION && GROUP_SCALES_WITH_ZP
    output_scale[scale_idx + 1] = zp;
#elif ASYMMETRIC_QUANTIZATION
    #if OUTPUT2_IS_FP
        output_zp[scale_idx] = zp;
    #elif UNSIGNED_OUTPUT
        output_zp[scale_idx] = convert_uchar_rte(zp);
    #else
        output_zp[scale_idx] = convert_char_rte(zp);
    #endif
#endif
}
