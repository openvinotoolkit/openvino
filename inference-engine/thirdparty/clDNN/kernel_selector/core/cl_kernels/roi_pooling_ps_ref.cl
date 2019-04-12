// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/common.cl"
#include "include/data_types.cl"

// Each RoI is described by 5 elements [batch_id xmin ymin xmax ymax]
#define ROI_NUM_ELEMENTS 5

#define COORD_T float
#define ACCUM_T float

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))

KERNEL(roi_pooling_ps_gpu)(const __global INPUT0_TYPE * src_data,
                                 __global OUTPUT_TYPE * dst_data,
                           const __global INPUT1_TYPE * src_rois)
{
    const size_t i = get_global_id(0);

    const uint x = i % OUTPUT_SIZE_X;
    const uint y = i / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    const uint c = i / OUTPUT_SIZE_X / OUTPUT_SIZE_Y % OUTPUT_FEATURE_NUM;
    const uint r = i / OUTPUT_SIZE_X / OUTPUT_SIZE_Y / OUTPUT_FEATURE_NUM % OUTPUT_ROI_NUM;

    const __global INPUT1_TYPE* roi_ptr = &src_rois[INPUT1_BATCH_PITCH * r];
    const int src_batch_idx = (int)(roi_ptr[0]);

#if BILINEAR_POOLING

    COORD_T roi_start_w = roi_ptr[1] * SPATIAL_SCALE;
    COORD_T roi_start_h = roi_ptr[2] * SPATIAL_SCALE;
    COORD_T roi_end_w   = roi_ptr[3] * SPATIAL_SCALE;
    COORD_T roi_end_h   = roi_ptr[4] * SPATIAL_SCALE;

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
            const __global INPUT0_TYPE* data = src_data + INPUT0_OFFSET + src_batch_idx*INPUT0_BATCH_PITCH + INPUT0_FEATURE_PITCH*gc;
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

                ACCUM_T top_left     = (ACCUM_T)data[top_y_index*INPUT0_Y_PITCH + left_x_index*INPUT0_X_PITCH];
                ACCUM_T top_right    = (ACCUM_T)data[top_y_index*INPUT0_Y_PITCH + right_x_index*INPUT0_X_PITCH];
                ACCUM_T bottom_left  = (ACCUM_T)data[bottom_y_index*INPUT0_Y_PITCH + left_x_index*INPUT0_X_PITCH];
                ACCUM_T bottom_right = (ACCUM_T)data[bottom_y_index*INPUT0_Y_PITCH + right_x_index*INPUT0_X_PITCH];

                ACCUM_T top    = top_left + (top_right - top_left) * (in_x - left_x_index);
                ACCUM_T bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

                res += top + (bottom - top) * (in_y - top_y_index);
            }
        }
    }

    res /= (SPATIAL_BINS_Y*SPATIAL_BINS_X);
#elif AVG_POOLING
    const uint work_c = x + POOLED_WIDTH * (y + POOLED_HEIGHT * c);
    const __global INPUT0_TYPE* data = src_data + INPUT0_OFFSET + src_batch_idx*INPUT0_BATCH_PITCH + INPUT0_FEATURE_PITCH*work_c;

    const COORD_T roi_x  = (COORD_T)(round(roi_ptr[1]) + 0.f) * SPATIAL_SCALE;
    const COORD_T roi_y  = (COORD_T)(round(roi_ptr[2]) + 0.f) * SPATIAL_SCALE;
    const COORD_T roi_x1 = (COORD_T)(round(roi_ptr[3]) + 1.f) * SPATIAL_SCALE;
    const COORD_T roi_y1 = (COORD_T)(round(roi_ptr[4]) + 1.f) * SPATIAL_SCALE;

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
            INPUT0_TYPE val = data[xx*INPUT0_X_PITCH + yy*INPUT0_Y_PITCH];
            res += (ACCUM_T)val;
        }
    }

    const COORD_T area = (y_after - y_begin) * (x_after - x_begin);
    if (area)
        res /= area;

#else
#error "Unsupported pooling mode"
#endif
    const uint output_offset = OUTPUT_OFFSET + x*OUTPUT_X_PITCH + y*OUTPUT_Y_PITCH + c*OUTPUT_FEATURE_PITCH + r*OUTPUT_ROI_PITCH;
    dst_data[output_offset] = ACTIVATION((OUTPUT_TYPE)(res), NL_M, NL_N);
}
