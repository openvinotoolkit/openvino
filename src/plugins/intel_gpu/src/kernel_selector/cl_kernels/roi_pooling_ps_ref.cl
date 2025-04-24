// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// Each RoI is described by 5 elements [batch_id xmin ymin xmax ymax]
#define ROI_NUM_ELEMENTS 5

#define COORD_T float
#define ACCUM_T float

ACCUM_T FUNC(bilinear_interp)(
      const __global INPUT0_TYPE * data
    , const int src_batch_idx
    , const int cc
    , const COORD_T x
    , const COORD_T y
    , const int width
    , const int height)
{
    int x1 = floor(x);
    int x2 = ceil(x);
    int y1 = floor(y);
    int y2 = ceil(y);
    COORD_T dist_x = (COORD_T)(x - x1);
    COORD_T dist_y = (COORD_T)(y - y1);
    COORD_T val1 = data[INPUT0_GET_INDEX(src_batch_idx, cc, y1, x1)];
    COORD_T val2 = data[INPUT0_GET_INDEX(src_batch_idx, cc, y2, x1)];
    COORD_T val21 = data[INPUT0_GET_INDEX(src_batch_idx, cc, y1, x2)];
    COORD_T val22 = data[INPUT0_GET_INDEX(src_batch_idx, cc, y2, x2)];
    ACCUM_T val = (1 - dist_x) * (1 - dist_y) * val1 + (1 - dist_x) * dist_y * val2
                  + dist_x * (1 - dist_y) * val21 + dist_x * dist_y * val22;
    return val;
}

KERNEL(roi_pooling_ps_gpu)(
      const __global INPUT0_TYPE * src_data
    , __global OUTPUT_TYPE * dst_data
    , const __global INPUT1_TYPE * src_rois
#ifdef DEFORMABLE_BILINEAR_POOLING
#if !NO_TRANS
    , const __global INPUT2_TYPE * trans_data
#endif
#endif
    )
{
    const size_t i = get_global_id(0);

    const uint x = i % OUTPUT_SIZE_X;
    const uint y = i / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    const uint c = i / OUTPUT_SIZE_X / OUTPUT_SIZE_Y % OUTPUT_FEATURE_NUM;
    const uint r = i / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_FEATURE_NUM % OUTPUT_BATCH_NUM;

    const int src_batch_idx = src_rois[INPUT1_GET_INDEX(r, 0, 0, 0)];

#if defined DEFORMABLE_BILINEAR_POOLING
    ACCUM_T res = 0.0f;

    COORD_T roi_start_w = (COORD_T)round(src_rois[INPUT1_GET_INDEX(r, 1, 0, 0)]) * SPATIAL_SCALE - 0.5;
    COORD_T roi_start_h = (COORD_T)round(src_rois[INPUT1_GET_INDEX(r, 2, 0, 0)]) * SPATIAL_SCALE - 0.5;
    COORD_T roi_end_w   = (COORD_T)round((src_rois[INPUT1_GET_INDEX(r, 3, 0, 0)]) + 1.0) * SPATIAL_SCALE - 0.5;
    COORD_T roi_end_h   = (COORD_T)round((src_rois[INPUT1_GET_INDEX(r, 4, 0, 0)]) + 1.0) * SPATIAL_SCALE - 0.5;

    COORD_T roi_width  = max((COORD_T)(roi_end_w - roi_start_w), 0.1f);
    COORD_T roi_height = max((COORD_T)(roi_end_h - roi_start_h), 0.1f);

    COORD_T bin_size_w = roi_width / (COORD_T)(POOLED_WIDTH);
    COORD_T bin_size_h = roi_height / (COORD_T)(POOLED_HEIGHT);

    COORD_T sub_bin_size_w = bin_size_w / (COORD_T)(SPATIAL_BINS_X);
    COORD_T sub_bin_size_h = bin_size_h / (COORD_T)(SPATIAL_BINS_Y);

    const int part_width = floor((COORD_T)x / POOLED_WIDTH * PART_SIZE + FLT_EPSILON);
    const int part_height = floor((COORD_T)y / POOLED_HEIGHT * PART_SIZE + FLT_EPSILON);

#if NO_TRANS
    const int num_classes = 1;
    const int channels_per_class = OUTPUT_FEATURE_NUM;
    const int class_idx = c / channels_per_class;

    COORD_T trans_x = 0;
    COORD_T trans_y = 0;
#else
    const int num_classes = INPUT2_FEATURE_NUM / 2;
    const int channels_per_class = OUTPUT_FEATURE_NUM / num_classes;
    const int class_idx = (c / channels_per_class) * 2;

    COORD_T trans_x = trans_data[INPUT2_GET_INDEX(r, class_idx, part_height, part_width)] * TRANS_STD;
    COORD_T trans_y = trans_data[INPUT2_GET_INDEX(r, class_idx + 1, part_height, part_width)] * TRANS_STD;
#endif

    COORD_T wstart = x * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;

    COORD_T hstart = y * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    ACCUM_T sum = 0;
    int count = 0;
    int gw = floor((COORD_T)x * GROUP_SIZE / POOLED_WIDTH);
    gw = min(max((COORD_T)x, 0.f), GROUP_SIZE - 1.f);
    int gh = floor((COORD_T)y * GROUP_SIZE / POOLED_HEIGHT);
    gh = min(max((COORD_T)y, 0.f), GROUP_SIZE - 1.f);

    for (int ih = 0; ih < SPATIAL_BINS_Y; ih++) {
        for (int iw = 0; iw < SPATIAL_BINS_X; iw++) {
            COORD_T w = wstart + iw * sub_bin_size_w;
            COORD_T h = hstart + ih * sub_bin_size_h;
            if (w < -0.5 || w > INPUT0_SIZE_X - 0.5 || h < -0.5 || h > INPUT0_SIZE_Y - 0.5) {
                continue;
            }
            w = min(max((COORD_T)w, 0.f), INPUT0_SIZE_X - 1.f);
            h = min(max((COORD_T)h, 0.f), INPUT0_SIZE_Y - 1.f);
            int cc = (c * GROUP_SIZE + gh) * GROUP_SIZE + gw;
            ACCUM_T val = FUNC_CALL(bilinear_interp)(src_data, src_batch_idx, cc, w, h, INPUT0_SIZE_X, INPUT0_SIZE_Y);
            sum += val;
            count++;
        }
    }

    res = count == 0 ? 0 : sum / count;
#elif defined BILINEAR_POOLING
    COORD_T roi_start_w = src_rois[INPUT1_GET_INDEX(r, 1, 0, 0)] * SPATIAL_SCALE;
    COORD_T roi_start_h = src_rois[INPUT1_GET_INDEX(r, 2, 0, 0)] * SPATIAL_SCALE;
    COORD_T roi_end_w   = src_rois[INPUT1_GET_INDEX(r, 3, 0, 0)] * SPATIAL_SCALE;
    COORD_T roi_end_h   = src_rois[INPUT1_GET_INDEX(r, 4, 0, 0)] * SPATIAL_SCALE;

    COORD_T roi_height = (roi_end_h - roi_start_h);
    COORD_T roi_width  = (roi_end_w - roi_start_w);

    ACCUM_T res = 0.0f;

    for (int bin_y = 0; bin_y < SPATIAL_BINS_Y; bin_y++)
    {
        for (int bin_x = 0; bin_x < SPATIAL_BINS_X; bin_x++)
        {
            COORD_T box_xmin = roi_start_w + (bin_x + 0) * (roi_width / SPATIAL_BINS_X);
            COORD_T box_xmax = roi_start_w + (bin_x + 1) * (roi_width / SPATIAL_BINS_X);
            COORD_T box_ymin = roi_start_h + (bin_y + 0) * (roi_height / SPATIAL_BINS_Y);
            COORD_T box_ymax = roi_start_h + (bin_y + 1) * (roi_height / SPATIAL_BINS_Y);

            const uint gc = c + (bin_y*SPATIAL_BINS_X + bin_x)*OUTPUT_FEATURE_NUM;
            COORD_T height_scale = POOLED_HEIGHT > 1 ? (box_ymax - box_ymin) * (INPUT0_SIZE_Y - 1) / (POOLED_HEIGHT - 1)
                                                     : 0.0f;
            COORD_T width_scale = POOLED_WIDTH > 1 ? (box_xmax - box_xmin) * (INPUT0_SIZE_X - 1) / (POOLED_WIDTH - 1)
                                                   : 0.0f;

            float in_y = POOLED_HEIGHT > 1 ? (y * height_scale + box_ymin * (INPUT0_SIZE_Y - 1))
                                           : 0.5f * (box_ymin + box_ymax) * (INPUT0_SIZE_Y - 1);
            float in_x = POOLED_WIDTH > 1 ? (x * width_scale + box_xmin * (INPUT0_SIZE_X - 1))
                                          : 0.5f * (box_xmin + box_xmax) * (INPUT0_SIZE_X - 1);

            if (!(in_y < 0 || in_y > (COORD_T)(INPUT0_SIZE_Y - 1) ||
                  in_x < 0 || in_x > (COORD_T)(INPUT0_SIZE_X - 1) || src_batch_idx == -1))
            {
                int top_y_index    = (int)(floor(in_y));
                int bottom_y_index = (int)(min(ceil(in_y), (COORD_T)INPUT0_SIZE_Y - 1));
                int left_x_index   = (int)(floor(in_x));
                int right_x_index  = (int)(min(ceil(in_x), (COORD_T)INPUT0_SIZE_X - 1));

                ACCUM_T top_left     = (ACCUM_T)src_data[INPUT0_GET_INDEX(src_batch_idx, gc, top_y_index, left_x_index)];
                ACCUM_T top_right    = (ACCUM_T)src_data[INPUT0_GET_INDEX(src_batch_idx, gc, top_y_index, right_x_index)];
                ACCUM_T bottom_left  = (ACCUM_T)src_data[INPUT0_GET_INDEX(src_batch_idx, gc, bottom_y_index, left_x_index)];
                ACCUM_T bottom_right = (ACCUM_T)src_data[INPUT0_GET_INDEX(src_batch_idx, gc, bottom_y_index, right_x_index)];

                ACCUM_T top    = top_left + (top_right - top_left) * (in_x - left_x_index);
                ACCUM_T bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

                res += top + (bottom - top) * (in_y - top_y_index);
            }
        }
    }

    res /= (SPATIAL_BINS_Y*SPATIAL_BINS_X);
#elif AVG_POOLING
    const uint work_c = x + POOLED_WIDTH * (y + POOLED_HEIGHT * c);

    const COORD_T roi_x  = (COORD_T)(round(src_rois[INPUT1_GET_INDEX(r, 1, 0, 0)]) + 0.f) * SPATIAL_SCALE;
    const COORD_T roi_y  = (COORD_T)(round(src_rois[INPUT1_GET_INDEX(r, 2, 0, 0)]) + 0.f) * SPATIAL_SCALE;
    const COORD_T roi_x1 = (COORD_T)(round(src_rois[INPUT1_GET_INDEX(r, 3, 0, 0)]) + 1.f) * SPATIAL_SCALE;
    const COORD_T roi_y1 = (COORD_T)(round(src_rois[INPUT1_GET_INDEX(r, 4, 0, 0)]) + 1.f) * SPATIAL_SCALE;

    // The final coordinate is within the ROI and malformed dimensions are treated as 1
    const COORD_T roi_w = max(roi_x1 - roi_x, .1f);
    const COORD_T roi_h = max(roi_y1 - roi_y, .1f);

    const COORD_T dx_begin = (x + 0) * (COORD_T)(roi_w / POOLED_WIDTH);
    const COORD_T dy_begin = (y + 0) * (COORD_T)(roi_h / POOLED_HEIGHT);
    const COORD_T dx_after = (x + 1) * (COORD_T)(roi_w / POOLED_WIDTH);
    const COORD_T dy_after = (y + 1) * (COORD_T)(roi_h / POOLED_HEIGHT);

    // clamp in case roi_x or roi_y were unreasonable
    const int x_begin = CLAMP(floor(roi_x + dx_begin), 0, INPUT0_SIZE_X);
    const int y_begin = CLAMP(floor(roi_y + dy_begin), 0, INPUT0_SIZE_Y);
    const int x_after = CLAMP(ceil(roi_x + dx_after), 0, INPUT0_SIZE_X);
    const int y_after = CLAMP(ceil(roi_y + dy_after), 0, INPUT0_SIZE_Y);

    ACCUM_T res = 0.0f;

    for (int yy = y_begin; yy < y_after; ++yy)
    {
        for (int xx = x_begin; xx < x_after; ++xx)
        {
            INPUT0_TYPE val = src_data[INPUT0_GET_INDEX(src_batch_idx, work_c, yy, xx)];
            res += (ACCUM_T)val;
        }
    }

    const COORD_T area = (y_after - y_begin) * (x_after - x_begin);
    if (area)
        res /= area;

#else
#error "Unsupported pooling mode"
#endif
    const uint output_offset = OUTPUT_GET_INDEX(r, c, y, x);
    dst_data[output_offset] = ACTIVATION((OUTPUT_TYPE)(res), ACTIVATION_PARAMS);
}
