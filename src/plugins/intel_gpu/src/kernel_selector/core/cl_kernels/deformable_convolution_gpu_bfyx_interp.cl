// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(deformable_convolution_gpu_bfyx_interp)(
    const __global INPUT0_TYPE* data,
    const __global INPUT1_TYPE* trans,
#if DEFORMABLE_MASK_ENABLED
    const __global INPUT2_TYPE* mask,
#endif
    __global OUTPUT_TYPE* output)
{
    const int xy = get_global_id(0);
    const int x = xy % OUTPUT_SIZE_X;
    const int y = xy / OUTPUT_SIZE_X;
    const int dg = (uint)get_global_id(1) % DEFORMABLE_GROUPS;
    const int b  = (uint)get_global_id(1) / DEFORMABLE_GROUPS;
    const int kw = (uint)get_global_id(2) % FILTER_SIZE_X;
    const int kh = (uint)get_global_id(2) / FILTER_SIZE_X;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const int channel_per_deformable_group = INPUT0_FEATURE_NUM / DEFORMABLE_GROUPS;

    const int input_offset_x = input_x + kw * DILATION_SIZE_X;
    const int input_offset_y = input_y + kh * DILATION_SIZE_Y;
    
#if DEFORMABLE_MASK_ENABLED
    const int dg_size = dg * FILTER_SIZE_Y * FILTER_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const int trans_offset = b * INPUT1_BATCH_PITCH + 2 * dg_size;
    const int filter_part_offset = kh * FILTER_SIZE_X + kw;
    const int trans_x_idx = ((2 * filter_part_offset + 1) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
    const int trans_y_idx = (2 * filter_part_offset * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
    const int mask_offset = b * INPUT2_BATCH_PITCH + dg_size;
    const int mask_idx = (filter_part_offset * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
#else
    const int trans_offset = b * INPUT1_BATCH_PITCH +
                             dg * 2 * FILTER_SIZE_Y * FILTER_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const int trans_x_idx = ((2 * (kh * FILTER_SIZE_X + kw) + 1) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
    const int trans_y_idx = ((2 * (kh * FILTER_SIZE_X + kw)) * OUTPUT_SIZE_Y + y) * OUTPUT_SIZE_X + x;
#endif
    float transformed_x = input_offset_x + (float) trans[trans_offset + trans_x_idx];
    float transformed_y = input_offset_y + (float) trans[trans_offset + trans_y_idx];
#if BILINEAR_INTERPOLATION_PAD
    const bool x_is_out_of_boundaries = (int)transformed_x >= INPUT0_SIZE_X || (int)transformed_x <= -1;
    const bool y_is_out_of_boundaries = (int)transformed_y >= INPUT0_SIZE_Y || (int)transformed_y <= -1;
#else
    const bool y_is_out_of_boundaries = transformed_y >= INPUT0_SIZE_Y || transformed_y < 0;
    const bool x_is_out_of_boundaries = transformed_x >= INPUT0_SIZE_X || transformed_x < 0;
#endif

    int top_y_index    = (int)(floor(transformed_y));
    int left_x_index   = (int)(floor(transformed_x));
#if BILINEAR_INTERPOLATION_PAD
    int bottom_y_index = top_y_index + 1;
    int right_x_index  = left_x_index + 1;
#else
    int bottom_y_index = (int)(min(ceil(transformed_y), (float)INPUT0_SIZE_Y - 1));
    int right_x_index  = (int)(min(ceil(transformed_x), (float)INPUT0_SIZE_X - 1));
#endif

    int oc = kh*FILTER_SIZE_X*INPUT0_FEATURE_NUM + kw*INPUT0_FEATURE_NUM + dg*channel_per_deformable_group;
    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, oc, y, x);
    if (!y_is_out_of_boundaries & !x_is_out_of_boundaries & xy < OUTPUT_SIZE_X*OUTPUT_SIZE_Y) {
        for (int c = 0; c < channel_per_deformable_group; ++c) {
            uint ifm = dg * channel_per_deformable_group + c;

#if BILINEAR_INTERPOLATION_PAD
            INPUT0_TYPE top_left     = top_y_index < 0 || left_x_index < 0 ? 0 : (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, left_x_index)];
            INPUT0_TYPE top_right    = top_y_index < 0 || right_x_index >= INPUT0_SIZE_X ? 0 :
                (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, right_x_index)];
            INPUT0_TYPE bottom_left  = bottom_y_index >= INPUT0_SIZE_Y || left_x_index < 0 ? 0 :
                (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, left_x_index)];
            INPUT0_TYPE bottom_right = bottom_y_index >= INPUT0_SIZE_Y || right_x_index >= INPUT0_SIZE_X ? 0 :
                (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, right_x_index)];
#else
            INPUT0_TYPE top_left     = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, left_x_index)];
            INPUT0_TYPE top_right    = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, top_y_index, right_x_index)];
            INPUT0_TYPE bottom_left  = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, left_x_index)];
            INPUT0_TYPE bottom_right = (INPUT0_TYPE)data[GET_DATA_INDEX(INPUT0, b, ifm, bottom_y_index, right_x_index)];
#endif

            INPUT0_TYPE top    = top_left + (top_right - top_left) * (transformed_x - left_x_index);
            INPUT0_TYPE bottom = bottom_left + (bottom_right - bottom_left) * (transformed_x - left_x_index);

#if DEFORMABLE_MASK_ENABLED
            output[dst_index + c*OUTPUT_FEATURE_PITCH] = (top + (bottom - top) * (transformed_y - top_y_index)) * mask[mask_offset + mask_idx];
#else
            output[dst_index + c*OUTPUT_FEATURE_PITCH] = top + (bottom - top) * (transformed_y - top_y_index);
#endif
        }
    } else if (xy < OUTPUT_SIZE_X*OUTPUT_SIZE_Y) {
        for (int c = 0; c < channel_per_deformable_group; ++c) {
            output[dst_index + c*OUTPUT_FEATURE_PITCH] = 0.0f;
        }
    }

}
