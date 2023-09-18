// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define NUM_ROIS      OUTPUT_BATCH_NUM
#define NUM_CHANNELS  INPUT0_FEATURE_NUM
#define POOLED_WIDTH  OUTPUT_SIZE_X
#define POOLED_HEIGHT OUTPUT_SIZE_Y

KERNEL(roi_align_ref)(const __global INPUT0_TYPE* src_data,
                            __global OUTPUT_TYPE* dst_data,
                      const __global INPUT1_TYPE* src_rois,
                      const __global INPUT2_TYPE* src_batches)
{
    const size_t i = get_global_id(0);

    const uint x = i % POOLED_WIDTH;
    const uint y = i / POOLED_WIDTH % POOLED_HEIGHT;
    const uint c = i / POOLED_WIDTH / POOLED_HEIGHT % NUM_CHANNELS;
    const uint r = i / POOLED_WIDTH / POOLED_HEIGHT / NUM_CHANNELS % NUM_ROIS;

    const __global INPUT1_TYPE* roi_ptr = &src_rois[INPUT1_GET_INDEX(r, 0, 0, 0)];

    // Get the batch index of feature map
    const uint b = (uint)src_batches[INPUT2_GET_INDEX(r, 0, 0, 0)];

    // Get ROI`s corners
    const INPUT1_TYPE x1 =
        (roi_ptr[0] + (INPUT1_TYPE)OFFSET_SRC) * (INPUT1_TYPE)SPATIAL_SCALE + (INPUT1_TYPE)OFFSET_DST;
    const INPUT1_TYPE y1 =
        (roi_ptr[1] + (INPUT1_TYPE)OFFSET_SRC) * (INPUT1_TYPE)SPATIAL_SCALE + (INPUT1_TYPE)OFFSET_DST;
    const INPUT1_TYPE x2 =
        (roi_ptr[2] + (INPUT1_TYPE)OFFSET_SRC) * (INPUT1_TYPE)SPATIAL_SCALE + (INPUT1_TYPE)OFFSET_DST;
    const INPUT1_TYPE y2 =
        (roi_ptr[3] + (INPUT1_TYPE)OFFSET_SRC) * (INPUT1_TYPE)SPATIAL_SCALE + (INPUT1_TYPE)OFFSET_DST;


    const INPUT1_TYPE roi_width = MAX(x2 - x1, (INPUT1_TYPE)MIN_SIZE);
    const INPUT1_TYPE roi_height = MAX(y2 - y1, (INPUT1_TYPE)MIN_SIZE);

    const INPUT1_TYPE bin_width = roi_width / POOLED_WIDTH;
    const INPUT1_TYPE bin_height = roi_height / POOLED_HEIGHT;

    const int sampling_ratio_x = SAMPLING_RATIO == 0 ? (int)ceil(bin_width) : SAMPLING_RATIO;
    const int sampling_ratio_y = SAMPLING_RATIO == 0 ? (int)ceil(bin_height) : SAMPLING_RATIO;

    const INPUT1_TYPE sample_distance_x = bin_width / (INPUT1_TYPE)sampling_ratio_x;
    const INPUT1_TYPE sample_distance_y = bin_height / (INPUT1_TYPE)sampling_ratio_y;

    OUTPUT_TYPE pooled_value = 0;
    for (unsigned int y_sample_ind = 0; y_sample_ind < sampling_ratio_y; y_sample_ind++) {
        INPUT1_TYPE sample_y =
            y1 + (INPUT1_TYPE)y * bin_height + sample_distance_y * ((INPUT1_TYPE)y_sample_ind + (INPUT1_TYPE)0.5f);
        for (unsigned int x_sample_ind = 0; x_sample_ind < sampling_ratio_x; x_sample_ind++) {
            INPUT1_TYPE sample_x =
                x1 + (INPUT1_TYPE)x * bin_width + sample_distance_x * ((INPUT1_TYPE)x_sample_ind + (INPUT1_TYPE)0.5f);
            unsigned int sample_y_low = 0;
            unsigned int sample_x_low = 0;
            unsigned int sample_y_high = 0;
            unsigned int sample_x_high = 0;
            INPUT1_TYPE weight_left = INPUT1_VAL_ZERO;
            INPUT1_TYPE weight_right = INPUT1_VAL_ZERO;
            INPUT1_TYPE weight_top = INPUT1_VAL_ZERO;
            INPUT1_TYPE weight_bottom = INPUT1_VAL_ZERO;
            if (sample_x >= -1.0 || sample_x <= INPUT0_SIZE_X || sample_y >= -1.0 || sample_y <= INPUT0_SIZE_Y) {
                sample_x = MAX(sample_x, INPUT1_VAL_ZERO);
                sample_y = MAX(sample_y, INPUT1_VAL_ZERO);

                sample_y_low = (unsigned int)sample_y;
                sample_x_low = (unsigned int)sample_x;

                if (sample_y_low >= INPUT0_SIZE_Y - 1) {
                    sample_y_high = sample_y_low = INPUT0_SIZE_Y - 1;
                    sample_y = (INPUT1_TYPE)sample_y_low;
                } else {
                    sample_y_high = sample_y_low + 1;
                }

                if (sample_x_low >= INPUT0_SIZE_X - 1) {
                    sample_x_high = sample_x_low = INPUT0_SIZE_X - 1;
                    sample_x = (INPUT1_TYPE)sample_x_low;
                } else {
                    sample_x_high = sample_x_low + 1;
                }

                // weight calculation for bilinear interpolation
                weight_top = sample_y - (INPUT1_TYPE)sample_y_low;
                weight_left = sample_x - (INPUT1_TYPE)sample_x_low;
                weight_bottom = INPUT1_VAL_ONE - weight_top;
                weight_right = INPUT1_VAL_ONE - weight_left;
            }

            const INPUT0_TYPE top_left = src_data[INPUT0_GET_INDEX(b, c, sample_y_low, sample_x_low)];
            const INPUT0_TYPE top_right = src_data[INPUT0_GET_INDEX(b, c, sample_y_low, sample_x_high)];
            const INPUT0_TYPE bottom_left = src_data[INPUT0_GET_INDEX(b, c, sample_y_high, sample_x_low)];
            const INPUT0_TYPE bottom_right = src_data[INPUT0_GET_INDEX(b, c, sample_y_high, sample_x_high)];

            const INPUT0_TYPE interpolated_value =
                weight_bottom * weight_right * top_left + weight_bottom * weight_left * top_right +
                weight_top * weight_right * bottom_left + weight_top * weight_left * bottom_right;

#if MAX_POOL
            pooled_value = MAX(pooled_value, interpolated_value);
#elif AVG_POOL
            pooled_value += interpolated_value;
#endif
        }
    }
#if AVG_POOL
    pooled_value /= sampling_ratio_x * sampling_ratio_x;
#endif

    dst_data[OUTPUT_GET_INDEX(r, c, y, x)] = ACTIVATION((OUTPUT_TYPE)pooled_value, ACTIVATION_PARAMS);
}
