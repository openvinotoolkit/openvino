// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

KERNEL(border_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#ifdef BEGIN_TYPE
    const __global BEGIN_TYPE* begin,
#endif
#ifdef END_TYPE
    const __global END_TYPE* end,
#endif
#ifdef BORDER_VALUE_TYPE
    const __global BORDER_VALUE_TYPE* border_value_param,
#endif
    __global OUTPUT_TYPE* output)
{
#if defined(BEGIN_TYPE) || defined(END_TYPE)
    uint pad_info_length = INPUT1_LENGTH;
#endif
#ifdef BEGIN_TYPE
    const int begin_b = begin[0];
    const int begin_f = begin[1];
    uint begin_offset = 2;
    #if INPUT0_DIMS == 6
    const int begin_w = begin[begin_offset];
    begin_offset += 1;
    #endif
    #if INPUT0_DIMS >= 5
    const int begin_z = begin[begin_offset];
    begin_offset += 1;
    #endif
    const int begin_y = (pad_info_length > begin_offset) ? begin[begin_offset] : 0;
    const int begin_x = (pad_info_length > (begin_offset + 1)) ? begin[begin_offset + 1] : 0;
#else
    const uint begin_b = LT_SIZES_BATCH_NUM;
    const uint begin_f = LT_SIZES_FEATURE_NUM;
    #if INPUT0_DIMS == 6
    const uint begin_w = LT_SIZES_SIZE_W;
    #endif
    #if INPUT0_DIMS >= 5
    const uint begin_z = LT_SIZES_SIZE_Z;
    #endif
    const uint begin_y = LT_SIZES_SIZE_Y;
    const uint begin_x = LT_SIZES_SIZE_X;
#endif

#ifdef END_TYPE
    const int end_b = end[0];
    const int end_f = end[1];
    uint end_offset = 2;
    #if INPUT0_DIMS == 6
    const int end_w = end[end_offset];
    end_offset += 1;
    #endif
    #if INPUT0_DIMS >= 5
    const int end_z = end[end_offset];
    end_offset += 1;
    #endif
    const int end_y = (pad_info_length > end_offset) ? end[end_offset] : 0;
    const int end_x = (pad_info_length > (end_offset + 1)) ? end[end_offset + 1] : 0;
#else
    const uint end_b = RB_SIZES_BATCH_NUM;
    const uint end_f = RB_SIZES_FEATURE_NUM;
    #if INPUT0_DIMS == 6
    const uint end_w = RB_SIZES_SIZE_W;
    #endif
    #if INPUT0_DIMS >= 5
    const uint end_z = RB_SIZES_SIZE_Z;
    #endif
    const uint end_y = RB_SIZES_SIZE_Y;
    const uint end_x = RB_SIZES_SIZE_X;
#endif

    // [CONSTEXPR]
    // Border sizes(left-top):
    const int blt_sb = begin_b;
    const int blt_sf = begin_f;
    const int blt_sy = begin_y;
    const int blt_sx = begin_x;
#if INPUT0_DIMS == 6
    const int blt_sw = begin_w;
#else
    const int blt_sw = 0;
#endif
#if INPUT0_DIMS >= 5
    const int blt_sz = begin_z;
#else
    const int blt_sz = 0;
#endif

    // Border sizes(right-bottom):
    const int brb_sb = end_b;
    const int brb_sf = end_f;
    const int brb_sy = end_y;
    const int brb_sx = end_x;
#if INPUT0_DIMS == 6
    const int brb_sw = end_w;
#else
    const int brb_sw = 0;
#endif
#if INPUT0_DIMS >= 5
    const int brb_sz = end_z;
#else
    const int brb_sz = 0;
#endif

    // Input sizes:
    const int in_sx = INPUT0_SIZE_X;
    const int in_sy = INPUT0_SIZE_Y;
    const int in_sz = INPUT0_SIZE_Z;
    const int in_sw = INPUT0_SIZE_W;
    const int in_sf = INPUT0_FEATURE_NUM;
    const int in_sb = INPUT0_BATCH_NUM;

    // Input limits (exclusive; tested on output position):
    const int in_lx = in_sx + blt_sx;
    const int in_ly = in_sy + blt_sy;
    const int in_lz = in_sz + blt_sz;
    const int in_lw = in_sw + blt_sw;
    const int in_lf = in_sf + blt_sf;
    const int in_lb = in_sb + blt_sb;

    const int out_xz = get_global_id(0);
    const int out_yw = get_global_id(1);
    const int out_fb = get_global_id(2);

    const int out_f  = out_fb % OUTPUT_FEATURE_NUM;
    const int out_b  = out_fb / OUTPUT_FEATURE_NUM;

    const int out_x  = out_xz % OUTPUT_SIZE_X;
    const int out_z  = out_xz / OUTPUT_SIZE_X;

    const int out_y  = out_yw % OUTPUT_SIZE_Y;
    const int out_w  = out_yw / OUTPUT_SIZE_Y;

#ifdef BORDER_TYPE_CONSTANT
    #ifdef BORDER_VALUE_TYPE
    INPUT0_TYPE in_val = TO_INPUT0_TYPE(border_value_param[0]);
    #else
    INPUT0_TYPE in_val = TO_INPUT0_TYPE(BORDER_VALUE);
    #endif

    if (out_x >= blt_sx & out_x < in_lx &
        out_y >= blt_sy & out_y < in_ly &
        out_z >= blt_sz & out_z < in_lz &
        out_w >= blt_sw & out_w < in_lw &
        out_f >= blt_sf & out_f < in_lf &
        out_b >= blt_sb & out_b < in_lb)
    {
        const int in_x = out_x - blt_sx;
        const int in_y = out_y - blt_sy;
        const int in_z = out_z - blt_sz;
        const int in_w = out_w - blt_sw;
        const int in_f = out_f - blt_sf;
        const int in_b = out_b - blt_sb;

        const int in_pos = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR in_b, in_f, in_w, in_z, in_y, in_x);
        in_val = input[in_pos];
    }
#elif defined BORDER_TYPE_EDGE
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? 0 : in_sx - 1);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? 0 : in_sy - 1);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? 0 : in_sz - 1);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? 0 : in_sw - 1);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? 0 : in_sf - 1);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? 0 : in_sb - 1);

    const uint in_pos = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR in_b, in_f, in_w, in_z, in_y, in_x);
    INPUT0_TYPE in_val = input[in_pos];
#elif defined BORDER_TYPE_MIRROR
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? blt_sx - 1 - out_x : in_sx + in_lx - 1 - out_x);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? blt_sy - 1 - out_y : in_sy + in_ly - 1 - out_y);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? blt_sz - 1 - out_z : in_sz + in_lz - 1 - out_z);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? blt_sw - 1 - out_w : in_sw + in_lw - 1 - out_w);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? blt_sf - 1 - out_f : in_sf + in_lf - 1 - out_f);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? blt_sb - 1 - out_b : in_sb + in_lb - 1 - out_b);

    const uint in_pos = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR in_b, in_f, in_w, in_z, in_y, in_x);
    INPUT0_TYPE in_val = input[in_pos];
#elif defined BORDER_TYPE_MIRROR_101
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? blt_sx - out_x : in_sx + in_lx - 2 - out_x);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? blt_sy - out_y : in_sy + in_ly - 2 - out_y);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? blt_sz - out_z : in_sz + in_lz - 2 - out_z);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? blt_sw - out_w : in_sw + in_lw - 2 - out_w);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? blt_sf - out_f : in_sf + in_lf - 2 - out_f);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? blt_sb - out_b : in_sb + in_lb - 2 - out_b);

    const uint in_pos = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR in_b, in_f, in_w, in_z, in_y, in_x);
    INPUT0_TYPE in_val = input[in_pos];
#else
    #error Unsupported border type.
#endif

    const int out_pos = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_w, out_z, out_y, out_x);
    output[out_pos] = in_val;
}
