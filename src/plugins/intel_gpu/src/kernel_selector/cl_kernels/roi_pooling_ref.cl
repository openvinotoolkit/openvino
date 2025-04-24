// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

/****************************************************************************
 *                                                                          *
 *                               Utility Defines                            *
 *                                                                          *
 ***************************************************************************/

// Each RoI is described by 5 elements, the first one being unused. This is
// required for the kernel to have the same API as other implmentations.
#define ROI_NUM_ELEMENTS 5

#define SRC_W INPUT0_SIZE_X
#define SRC_H INPUT0_SIZE_Y
#define DST_W POOLED_WIDTH
#define DST_H POOLED_HEIGHT
#define PITCH_ROI_R INPUT1_BATCH_PITCH

#define DST_C INPUT0_FEATURE_NUM

// Note: In the non-ROI_OLD case we keep the coordinates in float instead
//       of using UNIT_TYPE, since with FP16 we might actually lose some
//       precision in the coordinates, given a sufficiently large W or H.
#define COORD_T float
#define ACCUM_T float

#if INPUT1_FEATURE_NUM != ROI_NUM_ELEMENTS
#error - unknown ROI_POOLING kernel type
#endif

KERNEL(roi_pooling_gpu)
(
    const __global INPUT0_TYPE * src_data,
    __global OUTPUT_TYPE * dst_data,
    const __global INPUT1_TYPE * src_rois
)
{
    const size_t i = get_global_id(0);

    const uint x = i % DST_W;
    const uint y = i / DST_W % DST_H;
    const uint c = i / DST_W / DST_H % DST_C;
    const uint r = i / DST_W / DST_H / DST_C % OUTPUT_BATCH_NUM;
    // const uint b = i / DST_W / DST_H / DST_C / OUTPUT_ROI_NUM; - TODO: support batching correctly
    // Note: The rounding of the coordinates is done prior to the mul
    //       with SPATIAL_SCALE: It makes sense since the resolution of
    //       the pooled data is limited by its dimensions. (Is this clear?)

    const int src_batch_idx = src_rois[INPUT1_GET_INDEX(r, 0, 0, 0)];

#if BILINEAR_POOLING
    const uint output_offset = OUTPUT_GET_INDEX(r, c, y, x);
    COORD_T in_y;
    COORD_T in_x;

    COORD_T roi_start_w = src_rois[INPUT1_GET_INDEX(r, 1, 0, 0)];
    COORD_T roi_start_h = src_rois[INPUT1_GET_INDEX(r, 2, 0, 0)];
    COORD_T roi_end_w = src_rois[INPUT1_GET_INDEX(r, 3, 0, 0)];
    COORD_T roi_end_h = src_rois[INPUT1_GET_INDEX(r, 4, 0, 0)];

    COORD_T height_scale = (POOLED_HEIGHT > 1)
                               ? (roi_end_h - roi_start_h) * (SRC_H - 1.0f) / (COORD_T)(POOLED_HEIGHT - 1.0f)
                               : (COORD_T)(0);
    COORD_T width_scale =
        (POOLED_WIDTH > 1) ? (roi_end_w - roi_start_w) * (SRC_W - 1.0f) / (COORD_T)(POOLED_WIDTH - 1.0f) : (COORD_T)(0);
    if (POOLED_HEIGHT > 1) {
        in_y = (y == POOLED_HEIGHT - 1) ? (COORD_T)(SRC_H - 1.0f) * roi_end_h
                                        : y * height_scale + roi_start_h * (COORD_T)(SRC_H - 1.0f);
    } else {
        in_y = 0.5 * (roi_end_h + roi_start_h) * (COORD_T)(SRC_H - 1.0f);
    }
    if (POOLED_WIDTH > 1) {
        in_x = (x == POOLED_WIDTH - 1) ? (COORD_T)(SRC_W - 1.0f) * roi_end_w
                                       : x * width_scale + roi_start_w * (COORD_T)(SRC_W - 1.0f);
    } else {
        in_x = 0.5 * (roi_end_w + roi_start_w) * (COORD_T)(SRC_W - 1.0f);
    }

    if (in_y < 0 || in_y > (COORD_T)(SRC_H - 1) || in_x < 0 || in_x > (COORD_T)(SRC_W - 1) || src_batch_idx == -1) {
        dst_data[output_offset] = ACTIVATION((OUTPUT_TYPE)0, ACTIVATION_PARAMS);
        return;
    }

    int top_y_index = (int)(floor(in_y));
    int bottom_y_index = (int)(min(ceil(in_y), (COORD_T)SRC_H - 1));
    int left_x_index = (int)(floor(in_x));
    int right_x_index = (int)(min(ceil(in_x), (COORD_T)SRC_W - 1));

    ACCUM_T top_left = (ACCUM_T)src_data[INPUT0_GET_INDEX(src_batch_idx, c, top_y_index, left_x_index)];
    ACCUM_T top_right = (ACCUM_T)src_data[INPUT0_GET_INDEX(src_batch_idx, c, top_y_index, right_x_index)];
    ACCUM_T bottom_left = (ACCUM_T)src_data[INPUT0_GET_INDEX(src_batch_idx, c, bottom_y_index, left_x_index)];
    ACCUM_T bottom_right = (ACCUM_T)src_data[INPUT0_GET_INDEX(src_batch_idx, c, bottom_y_index, right_x_index)];

    ACCUM_T top = top_left + (top_right - top_left) * (in_x - left_x_index);
    ACCUM_T bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

    ACCUM_T res = top + (bottom - top) * (in_y - top_y_index);

    dst_data[output_offset] = ACTIVATION((OUTPUT_TYPE)res, ACTIVATION_PARAMS);
#else

    const int roi_x  = round(src_rois[INPUT1_GET_INDEX(r, 1, 0, 0)] * SPATIAL_SCALE);
    const int roi_y  = round(src_rois[INPUT1_GET_INDEX(r, 2, 0, 0)] * SPATIAL_SCALE);
    const int roi_x1 = round(src_rois[INPUT1_GET_INDEX(r, 3, 0, 0)] * SPATIAL_SCALE);
    const int roi_y1 = round(src_rois[INPUT1_GET_INDEX(r, 4, 0, 0)] * SPATIAL_SCALE);

    // The final coordinate is within the ROI and malformed dimensions are treated as 1
    const uint roi_w = max(roi_x1 - roi_x, 0) + 1;
    const uint roi_h = max(roi_y1 - roi_y, 0) + 1;

    // Note that when the "after" is rounded rounded up else we get the last cell,
    // instead of the cell beyond (For "symmetry").
    //
    // For ex. with src being a 6 cell row and dest being a 4 cell one:
    // >>> [((x + 0) * 6) // 4 for x in [0, 1, 2, 3]]   # "begin" values
    // [0, 1, 3, 4]                                     # as expected
    // >>> [((x + 1) * 6) // 4 for x in [0, 1, 2, 3]]   # "after" values
    // [1, 3, 4 ,6]                                     # [2, 3, 5, 6] expected!
    const int dx_begin = ((x + 0) * roi_w) / DST_W;
    const int dy_begin = ((y + 0) * roi_h) / DST_H;
    const int dx_after = ((x + 1) * roi_w + (DST_W - 1)) / DST_W;
    const int dy_after = ((y + 1) * roi_h + (DST_H - 1)) / DST_H;

    // clamp in case roi_x or roi_y were unreasonable
    const int x_begin = clamp(roi_x + dx_begin, 0, SRC_W);
    const int y_begin = clamp(roi_y + dy_begin, 0, SRC_H);
    const int x_after = clamp(roi_x + dx_after, 0, SRC_W);
    const int y_after = clamp(roi_y + dy_after, 0, SRC_H);


#if MAX_POOLING
    ACCUM_T res = x_begin < x_after && y_begin < y_after ? -FLT_MAX : 0;
#else
    ACCUM_T res = 0;
#endif

    for (int yy = y_begin; yy < y_after; ++yy)
    for (int xx = x_begin; xx < x_after; ++xx)
    {
        INPUT0_TYPE val = src_data[INPUT0_GET_INDEX(src_batch_idx, c, yy, xx)];
#if MAX_POOLING
        res = MAX(res, (ACCUM_T)val);
#else
        res = res + (ACCUM_T)val;
#endif
    }

#if (!MAX_POOLING)
    {
        const COORD_T area = (y_after - y_begin) * (x_after - x_begin);
        if (area) res /= area;
    }
#endif

    const uint output_offset = OUTPUT_GET_INDEX(r, c, y, x);
    dst_data[output_offset] = ACTIVATION((OUTPUT_TYPE)res, ACTIVATION_PARAMS);
#endif
}
