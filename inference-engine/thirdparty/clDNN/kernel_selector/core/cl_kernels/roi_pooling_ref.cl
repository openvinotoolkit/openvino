// Copyright (c) 2016-2017 Intel Corporation
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


#include "include/include_all.cl"


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

#if GROUP_SIZE == 0
#define DST_C INPUT0_FEATURE_NUM
#else
#define DST_C (GROUP_SIZE ? (INPUT0_FEATURE_NUM / GROUP_SIZE / GROUP_SIZE) : INPUT0_FEATURE_NUM)
#endif

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

/****************************************************************************
 *                                                                          *
 *                                RoI Pooling                               *
 *                                                                          *
 ***************************************************************************/

KERNEL(roi_pooling_gpu)
(
    const __global UNIT_TYPE * src_data,
    __global UNIT_TYPE * dst_data,
    const __global UNIT_TYPE * src_rois
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

    const __global UNIT_TYPE * roi_ptr = &src_rois[PITCH_ROI_R * r];
#if USE_OLD_SCALE_AND_ROUNDING
    const int roi_x  = round(roi_ptr[1] * SPATIAL_SCALE);
    const int roi_y  = round(roi_ptr[2] * SPATIAL_SCALE);
    const int roi_x1 = round(roi_ptr[3] * SPATIAL_SCALE);
    const int roi_y1 = round(roi_ptr[4] * SPATIAL_SCALE);

    // The final coordinate is within the ROI and malformed dimensions are treated as 1
    const uint roi_w = max(roi_x1 - roi_x, 0) + 1;
    const uint roi_h = max(roi_y1 - roi_y, 0) + 1;
#else
    const COORD_T roi_x  = (COORD_T)(round(roi_ptr[1]) + 0.f) * SPATIAL_SCALE;
    const COORD_T roi_y  = (COORD_T)(round(roi_ptr[2]) + 0.f) * SPATIAL_SCALE;
    const COORD_T roi_x1 = (COORD_T)(round(roi_ptr[3]) + 1.f) * SPATIAL_SCALE;
    const COORD_T roi_y1 = (COORD_T)(round(roi_ptr[4]) + 1.f) * SPATIAL_SCALE;

    // The final coordinate is within the ROI and malformed dimensions are treated as 1
    const COORD_T roi_w = max(roi_x1 - roi_x, .1f);
    const COORD_T roi_h = max(roi_y1 - roi_y, .1f);
#endif

    // Note that when the "after" is rounded rounded up else we get the last cell,
    // instead of the cell beyond (For "symmetry").
    //
    // For ex. with src being a 6 cell row and dest being a 4 cell one:
    // >>> [((x + 0) * 6) // 4 for x in [0, 1, 2, 3]]   # "begin" values
    // [0, 1, 3, 4]                                     # as expected
    // >>> [((x + 1) * 6) // 4 for x in [0, 1, 2, 3]]   # "after" values
    // [1, 3, 4 ,6]                                     # [2, 3, 5, 6] expected!
#if USE_OLD_SCALE_AND_ROUNDING
    const int dx_begin = ((x + 0) * roi_w) / DST_W;
    const int dy_begin = ((y + 0) * roi_h) / DST_H;
    const int dx_after = ((x + 1) * roi_w + (DST_W - 1)) / DST_W;
    const int dy_after = ((y + 1) * roi_h + (DST_H - 1)) / DST_H;

    // clamp in case roi_x or roi_y were unreasonable
    const int x_begin = clamp(roi_x + dx_begin, 0, SRC_W);
    const int y_begin = clamp(roi_y + dy_begin, 0, SRC_H);
    const int x_after = clamp(roi_x + dx_after, 0, SRC_W);
    const int y_after = clamp(roi_y + dy_after, 0, SRC_H);
#else
    const COORD_T dx_begin = (x + 0) * (COORD_T)(roi_w / DST_W);
    const COORD_T dy_begin = (y + 0) * (COORD_T)(roi_h / DST_H);
    const COORD_T dx_after = (x + 1) * (COORD_T)(roi_w / DST_W);
    const COORD_T dy_after = (y + 1) * (COORD_T)(roi_h / DST_H);

    // clamp in case roi_x or roi_y were unreasonable
    const int x_begin = CLAMP(floor(roi_x + dx_begin), 0, SRC_W);
    const int y_begin = CLAMP(floor(roi_y + dy_begin), 0, SRC_H);
    const int x_after = CLAMP(ceil(roi_x + dx_after), 0, SRC_W);
    const int y_after = CLAMP(ceil(roi_y + dy_after), 0, SRC_H);
#endif

#if GROUP_SIZE == 0
    const uint work_c = c;
#else

#if 0
    const COORD_T group_bin_w = (COORD_T)roi_w / DST_W;
    const COORD_T group_bin_h = (COORD_T)roi_h / DST_H;
    
    const uint group_x = CLAMP(x * group_bin_w, 0, GROUP_SIZE - 1);
    const uint group_y = CLAMP(y * group_bin_h, 0, GROUP_SIZE - 1);
#else
    const uint group_x = x;
    const uint group_y = y;
#endif

    const uint work_c = group_x + GROUP_SIZE * (group_y + GROUP_SIZE * c);
#endif

    const __global UNIT_TYPE * data = src_data + INPUT0_OFFSET + INPUT0_FEATURE_PITCH*work_c;

    ACCUM_T res = MAX_POOL && x_begin < x_after && y_begin < y_after ? UNIT_VAL_MIN : 0;

    for (int yy = y_begin; yy < y_after; ++yy)
    for (int xx = x_begin; xx < x_after; ++xx)
    {
        UNIT_TYPE val = data[xx*INPUT0_X_PITCH + yy*INPUT0_Y_PITCH];

        res = MAX_POOL ? MAX(res, (ACCUM_T)val) : res + (ACCUM_T)val;
    }

    if (!MAX_POOL)
    {
        //TODO(ruv): again, differs from the standard fixed size area (?)
        const COORD_T area = (y_after - y_begin) * (x_after - x_begin);
        if (area) res /= area;
    }

    const uint output_offset = OUTPUT_OFFSET + x*OUTPUT_X_PITCH + y*OUTPUT_Y_PITCH + c*OUTPUT_FEATURE_PITCH + r*OUTPUT_ROI_PITCH;
    dst_data[output_offset] = ACTIVATION((UNIT_TYPE)res, NL_M, NL_N);
} 