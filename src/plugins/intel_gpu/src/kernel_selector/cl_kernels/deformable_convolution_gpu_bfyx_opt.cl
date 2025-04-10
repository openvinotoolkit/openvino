// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/unit_type.cl"

#ifdef DEFORMABLE_CONV_STAGE_0

REQD_SUB_GROUP_SIZE(16)
KERNEL(deformable_convolution_gpu_bfyx_interp)(
    const __global INPUT0_TYPE* data,
    const __global INPUT1_TYPE* trans,
#if DEFORMABLE_MASK_ENABLED
    const __global INPUT2_TYPE* mask,
#endif
    __global INTERPOLATED_TYPE* output)
{
    const int xy = get_global_id(0);
    const int x = xy % INTERPOLATED_SIZE_X;
    const int y = xy / INTERPOLATED_SIZE_X;
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
    const int dg_size = dg * FILTER_SIZE_Y * FILTER_SIZE_X * INTERPOLATED_SIZE_Y * INTERPOLATED_SIZE_X;
    const int trans_offset = b * INPUT1_BATCH_PITCH + 2 * dg_size;
    const int filter_part_offset = kh * FILTER_SIZE_X + kw;
    const int trans_x_idx = ((2 * filter_part_offset + 1) * INTERPOLATED_SIZE_Y + y) * INTERPOLATED_SIZE_X + x;
    const int trans_y_idx = (2 * filter_part_offset * INTERPOLATED_SIZE_Y + y) * INTERPOLATED_SIZE_X + x;
    const int mask_offset = b * INPUT2_BATCH_PITCH + dg_size;
    const int mask_idx = (filter_part_offset * INTERPOLATED_SIZE_Y + y) * INTERPOLATED_SIZE_X + x;
#else
    const int trans_offset = b * INPUT1_BATCH_PITCH +
                             dg * 2 * FILTER_SIZE_Y * FILTER_SIZE_X * INTERPOLATED_SIZE_Y * INTERPOLATED_SIZE_X;
    const int trans_x_idx = ((2 * (kh * FILTER_SIZE_X + kw) + 1) * INTERPOLATED_SIZE_Y + y) * INTERPOLATED_SIZE_X + x;
    const int trans_y_idx = ((2 * (kh * FILTER_SIZE_X + kw)) * INTERPOLATED_SIZE_Y + y) * INTERPOLATED_SIZE_X + x;
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
    const uint dst_index = GET_DATA_INDEX(INTERPOLATED, b, oc, y, x);
    if (!y_is_out_of_boundaries & !x_is_out_of_boundaries & xy < INTERPOLATED_SIZE_X*INTERPOLATED_SIZE_Y) {
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
            output[dst_index + c*INTERPOLATED_FEATURE_PITCH] = (top + (bottom - top) * (transformed_y - top_y_index)) * mask[mask_offset + mask_idx];
#else
            output[dst_index + c*INTERPOLATED_FEATURE_PITCH] = top + (bottom - top) * (transformed_y - top_y_index);
#endif
        }
    } else if (xy < INTERPOLATED_SIZE_X*INTERPOLATED_SIZE_Y) {
        for (int c = 0; c < channel_per_deformable_group; ++c) {
            output[dst_index + c*INTERPOLATED_FEATURE_PITCH] = 0.0f;
        }
    }

}
#endif // DEFORMABLE_CONV_STAGE_0

#ifdef DEFORMABLE_CONV_STAGE_1
#define FEATURE_SLICE_SIZE 16

#define GET_WEI(filter, id) AS_TYPE(UNIT_TYPE, _sub_group_shuffle(AS_TYPE(UNIT_BLOCK_RW_TYPE, filter), id))

REQD_SUB_GROUP_SIZE(16)
KERNEL(deformable_convolution_gpu_bfyx_conv)(
    const __global INTERPOLATED_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
)
{
    const uint lid = get_sub_group_local_id();
    const uint x = ((uint)get_global_id(0) * X_BLOCK_SIZE + lid) % OUTPUT_SIZE_X;
    const uint y = ((uint)get_global_id(0) * X_BLOCK_SIZE + lid) / OUTPUT_SIZE_X;
    const uint f_block = get_group_id(1);
    const uint b = get_global_id(2);

    UNIT_TYPE dotProd[16] = { UNIT_VAL_ZERO };

    // Filter offset calculations:
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
    const uint filter_offset = f_block * filter_os_pitch;

    const uint input_offset = INTERPOLATED_OFFSET + b*INTERPOLATED_BATCH_PITCH + x*INTERPOLATED_X_PITCH + y*INTERPOLATED_Y_PITCH;
    const uint input_kh_pitch = FILTER_SIZE_X*INPUT_CHANNELS*INTERPOLATED_FEATURE_PITCH;
    const uint input_kw_pitch = INPUT_CHANNELS*INTERPOLATED_FEATURE_PITCH;

    for (uint kh = 0; kh < FILTER_SIZE_Y ; ++kh)
    {
        for (uint kw = 0; kw < FILTER_SIZE_X ; ++kw)
        {
            for (uint icb = 0; icb < (INPUT_CHANNELS + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE; ++icb)
            {
                UNIT_TYPE8 wei0 = UNIT_BLOCK_READ8(weights, filter_offset +
                                                            icb * filter_is_pitch +
                                                            kh * filter_y_pitch +
                                                            kw * filter_x_pitch);
                UNIT_TYPE8 wei1 = UNIT_BLOCK_READ8(weights, filter_offset +
                                                            icb * filter_is_pitch +
                                                            kh * filter_y_pitch +
                                                            kw * filter_x_pitch +
                                                            8 * filter_isv_pitch);

                UNIT_TYPE src[16];
                for (int ic = 0; ic < 16; ic++) {
                    if (icb*FEATURE_SLICE_SIZE + ic < INPUT_CHANNELS)
                        src[ic] = input[input_offset +
                                        kh*input_kh_pitch +
                                        kw*input_kw_pitch +
                                        (icb*FEATURE_SLICE_SIZE + ic)*INTERPOLATED_FEATURE_PITCH];
                    else
                        src[ic] = 0.0f;
                }

                for (int oc = 0; oc < 16; oc++) {
                    dotProd[oc] += src[0] * GET_WEI(wei0.s0, oc);
                    dotProd[oc] += src[1] * GET_WEI(wei0.s1, oc);
                    dotProd[oc] += src[2] * GET_WEI(wei0.s2, oc);
                    dotProd[oc] += src[3] * GET_WEI(wei0.s3, oc);
                    dotProd[oc] += src[4] * GET_WEI(wei0.s4, oc);
                    dotProd[oc] += src[5] * GET_WEI(wei0.s5, oc);
                    dotProd[oc] += src[6] * GET_WEI(wei0.s6, oc);
                    dotProd[oc] += src[7] * GET_WEI(wei0.s7, oc);
                    dotProd[oc] += src[8] * GET_WEI(wei1.s0, oc);
                    dotProd[oc] += src[9] * GET_WEI(wei1.s1, oc);
                    dotProd[oc] += src[10] * GET_WEI(wei1.s2, oc);
                    dotProd[oc] += src[11] * GET_WEI(wei1.s3, oc);
                    dotProd[oc] += src[12] * GET_WEI(wei1.s4, oc);
                    dotProd[oc] += src[13] * GET_WEI(wei1.s5, oc);
                    dotProd[oc] += src[14] * GET_WEI(wei1.s6, oc);
                    dotProd[oc] += src[15] * GET_WEI(wei1.s7, oc);
                }
            }
        }
    }

    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f_block*FEATURE_SLICE_SIZE, y, x);
    for (int oc = 0; oc < 16; oc++)
    {
#if BIAS_TERM
        const uint bias_index = f_block*FEATURE_SLICE_SIZE + oc;
        dotProd[oc] += (UNIT_TYPE)biases[bias_index];
#endif
        if ((uint)get_global_id(0) * X_BLOCK_SIZE + lid < OUTPUT_SIZE_X*OUTPUT_SIZE_Y && f_block*FEATURE_SLICE_SIZE + oc < OUTPUT_FEATURE_NUM)
            output[dst_index + oc*OUTPUT_FEATURE_PITCH] = ACTIVATION(dotProd[oc], ACTIVATION_PARAMS);
    }

}
#endif // DEFORMABLE_CONV_STAGE_1
