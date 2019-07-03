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

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(v,l,u) MAX((l),MIN((v),(u)))

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
    const uint r = i / DST_W / DST_H / DST_C % OUTPUT_ROI_NUM;
    // const uint b = i / DST_W / DST_H / DST_C / OUTPUT_ROI_NUM; - TODO: support batching correctly
    // Note: The rounding of the coordinates is done prior to the mul
    //       with SPATIAL_SCALE: It makes sense since the resolution of
    //       the pooled data is limited by its dimensions. (Is this clear?)

    const __global INPUT1_TYPE* roi_ptr = &src_rois[PITCH_ROI_R * r];

    const int src_batch_idx = (int)(roi_ptr[0]);

#if BILINEAR_POOLING
    const uint output_offset = OUTPUT_OFFSET + x*OUTPUT_X_PITCH + y*OUTPUT_Y_PITCH + c*OUTPUT_FEATURE_PITCH + r*OUTPUT_ROI_PITCH;

    COORD_T roi_start_w = roi_ptr[1];
    COORD_T roi_start_h = roi_ptr[2];
    COORD_T roi_end_w   = roi_ptr[3];
    COORD_T roi_end_h   = roi_ptr[4];

    COORD_T height_scale = (roi_end_h - roi_start_h) * (SRC_H - 1.0f) / (COORD_T)(POOLED_HEIGHT - 1.0f);
    COORD_T width_scale  = (roi_end_w - roi_start_w) * (SRC_W - 1.0f) / (COORD_T)(POOLED_WIDTH  - 1.0f);

    COORD_T in_y = y*height_scale + roi_start_h*(COORD_T)(SRC_H - 1.0f);
    COORD_T in_x = x*width_scale  + roi_start_w*(COORD_T)(SRC_W - 1.0f);

    if (in_y < 0 || in_y > (COORD_T)(SRC_H - 1) || in_x < 0 || in_x > (COORD_T)(SRC_W - 1) || src_batch_idx == -1) {
        dst_data[output_offset] = ACTIVATION((OUTPUT_TYPE)0, NL_M, NL_N);
        return;
    }

    int top_y_index    = (int)(floor(in_y));
    int bottom_y_index = (int)(min(ceil(in_y), (COORD_T)SRC_H - 1));
    int left_x_index   = (int)(floor(in_x));
    int right_x_index  = (int)(min(ceil(in_x), (COORD_T)SRC_W - 1));

    const __global INPUT0_TYPE* data = src_data + INPUT0_OFFSET + src_batch_idx*INPUT0_BATCH_PITCH + INPUT0_FEATURE_PITCH*c;

    ACCUM_T top_left     = (ACCUM_T)data[top_y_index*INPUT0_Y_PITCH + left_x_index*INPUT0_X_PITCH];
    ACCUM_T top_right    = (ACCUM_T)data[top_y_index*INPUT0_Y_PITCH + right_x_index*INPUT0_X_PITCH];
    ACCUM_T bottom_left  = (ACCUM_T)data[bottom_y_index*INPUT0_Y_PITCH + left_x_index*INPUT0_X_PITCH];
    ACCUM_T bottom_right = (ACCUM_T)data[bottom_y_index*INPUT0_Y_PITCH + right_x_index*INPUT0_X_PITCH];

    ACCUM_T top    = top_left + (top_right - top_left) * (in_x - left_x_index);
    ACCUM_T bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

    ACCUM_T res = top + (bottom - top) * (in_y - top_y_index);

    dst_data[output_offset] = ACTIVATION((OUTPUT_TYPE)res, NL_M, NL_N);
#else

    const int roi_x  = round(roi_ptr[1] * SPATIAL_SCALE);
    const int roi_y  = round(roi_ptr[2] * SPATIAL_SCALE);
    const int roi_x1 = round(roi_ptr[3] * SPATIAL_SCALE);
    const int roi_y1 = round(roi_ptr[4] * SPATIAL_SCALE);

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

    const __global INPUT0_TYPE* data = src_data + INPUT0_OFFSET + src_batch_idx*INPUT0_BATCH_PITCH + INPUT0_FEATURE_PITCH*c;

#if MAX_POOLING
    ACCUM_T res = x_begin < x_after && y_begin < y_after ? -FLT_MAX : 0;
#else
    ACCUM_T res = 0;
#endif

    for (int yy = y_begin; yy < y_after; ++yy)
    for (int xx = x_begin; xx < x_after; ++xx)
    {
        INPUT0_TYPE val = data[xx*INPUT0_X_PITCH + yy*INPUT0_Y_PITCH];
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

    const uint output_offset = OUTPUT_OFFSET + x*OUTPUT_X_PITCH + y*OUTPUT_Y_PITCH + c*OUTPUT_FEATURE_PITCH + r*OUTPUT_ROI_PITCH;
    dst_data[output_offset] = ACTIVATION((OUTPUT_TYPE)res, NL_M, NL_N);
#endif
}
