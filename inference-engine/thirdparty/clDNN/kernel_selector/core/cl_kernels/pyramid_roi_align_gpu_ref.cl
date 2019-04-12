// Copyright (c) 2018 Intel Corporation
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

#define META_OFFSET_X                   4
#define META_OFFSET_Y                   5

#define SIZE_TAB_PARAMETERS             4

struct Parameters
{
    int h_source, w_source, f_Size, x_Size, y_Size, offset;
};

__constant struct Parameters parameters [SIZE_TAB_PARAMETERS] =
        {
            { INPUT2_SIZE_Y, INPUT2_SIZE_X, INPUT2_FEATURE_PITCH, INPUT2_X_PITCH, INPUT2_Y_PITCH, INPUT2_OFFSET },
            { INPUT3_SIZE_Y, INPUT3_SIZE_X, INPUT3_FEATURE_PITCH, INPUT3_X_PITCH, INPUT3_Y_PITCH, INPUT3_OFFSET },
            { INPUT4_SIZE_Y, INPUT4_SIZE_X, INPUT4_FEATURE_PITCH, INPUT4_X_PITCH, INPUT4_Y_PITCH, INPUT4_OFFSET },
            { INPUT5_SIZE_Y, INPUT5_SIZE_X, INPUT5_FEATURE_PITCH, INPUT5_X_PITCH, INPUT5_Y_PITCH, INPUT5_OFFSET }
        };


KERNEL(pyramidROIAlign_gpu_ref)(
    const __global INPUT0_TYPE *boxes,
    const __global INPUT1_TYPE *image_meta,
    const __global INPUT2_TYPE *P2,
    const __global INPUT3_TYPE *P3,
    const __global INPUT4_TYPE *P4,
    const __global INPUT5_TYPE *P5,
    const __global INPUT6_TYPE *dim,
    __global OUTPUT_TYPE *output)
{
    // [CONSTEXPR]:
    const uint kerNum = (uint) get_global_id(0);

    const __global float *feature_map_Ptr[SIZE_TAB_PARAMETERS];
    int f_Size;

    INPUT1_TYPE img_dim_X = image_meta[GET_DATA_INDEX(INPUT1, 0, 0, 0, META_OFFSET_X)];
    INPUT1_TYPE img_dim_Y = image_meta[GET_DATA_INDEX(INPUT1, 0, 0, 0, META_OFFSET_Y)];

    INPUT1_TYPE image_area = img_dim_X * img_dim_Y;
    INPUT1_TYPE scale = sqrt(image_area) / 224.0;

    INPUT0_TYPE hU = boxes[GET_DATA_INDEX(INPUT0, 0, 0, kerNum, 2)];
    INPUT0_TYPE hL = boxes[GET_DATA_INDEX(INPUT0, 0, 0, kerNum, 0)];
    INPUT0_TYPE h = hU - hL;
    INPUT0_TYPE wU = boxes[GET_DATA_INDEX(INPUT0, 0, 0, kerNum, 3)];
    INPUT0_TYPE wL = boxes[GET_DATA_INDEX(INPUT0, 0, 0, kerNum, 1)];
    INPUT0_TYPE w = wU - wL;

    int roi_level = (int)round(log2(sqrt(h*w) * scale));

    // 0 <= roi_level <= 3
    roi_level = min(3, max(0, 2 + roi_level));

    feature_map_Ptr[0] = P2;
    feature_map_Ptr[1] = P3;
    feature_map_Ptr[2] = P4;
    feature_map_Ptr[3] = P5;

    f_Size = parameters[roi_level].f_Size;

    //calculate cooficients for transformation
    INPUT0_TYPE y1 = hL * (parameters[roi_level].h_source - 1);
    INPUT0_TYPE x1 = wL * (parameters[roi_level].w_source - 1);
    INPUT0_TYPE y2 = hU * (parameters[roi_level].h_source - 1);
    INPUT0_TYPE x2 = wU * (parameters[roi_level].w_source - 1);
    INPUT0_TYPE deltaX = (x2 - x1) / (OUTPUT_SIZE_X - 1);
    INPUT0_TYPE deltaY = (y2 - y1) / (OUTPUT_SIZE_Y - 1);
    INPUT0_TYPE y = y1;

   //transformation
    for (int i = 0; i < OUTPUT_SIZE_Y; ++i) //loop by 'y' coordinate
    {
        int ya = (int)floor(y);
        int yb = (int)ceil(y);

        if (ya < 0) ya = 0;
        if (yb >= parameters[roi_level].h_source) yb = parameters[roi_level].h_source - 1;
        if (yb - ya == 0)
        {
            if (yb + 2 < parameters[roi_level].h_source) ++yb;
            else --ya;
        }

        INPUT0_TYPE x = x1;

        for (int j = 0; j < OUTPUT_SIZE_X; ++j) //loop by 'x' coordinate
        {
            int xa = (int)floor(x);
            int xb = (int)ceil(x);
            if (xa < 0) xa = 0;
            if (xb >= parameters[roi_level].w_source) xb = parameters[roi_level].w_source - 1;
            if (xb - xa == 0)
            {
                if (xb + 2 < parameters[roi_level].w_source) ++xb;
                else --xa;
            }

    /* BILINEAR TRANSFORMATION
         (xa,yb,f3)*---------------------------------*(xb,yb,f2)
                   |                                 |
                   |          *(x,y)                 |
                   |                                 |
         (xa,ya,f0)*---------------------------------*(xb,ya,f1)
   */
            //cooficients for bilinear transformation
            INPUT0_TYPE a = yb - y;
            INPUT0_TYPE b = y - ya;
            INPUT0_TYPE c = xb - x;
            INPUT0_TYPE d = x - xa;

            /*#define GET_DATA_INDEX(prefix, b, f, y, x)  \
                CAT(prefix, _OFFSET) +                  \
                (x)*CAT(prefix, _X_PITCH) +             \
                (y)*CAT(prefix, _Y_PITCH) +             \
                (f)*CAT(prefix, _FEATURE_PITCH) +       \
                (b)*CAT(prefix, _BATCH_PITCH)

            For P2, P3, P4, P5 batch size is always 0 */

            size_t f0Ind = parameters[roi_level].offset + parameters[roi_level].y_Size * ya + parameters[roi_level].x_Size * xa;
            size_t f1Ind = parameters[roi_level].offset + parameters[roi_level].y_Size * ya + parameters[roi_level].x_Size * xb;
            size_t f2Ind = parameters[roi_level].offset + parameters[roi_level].y_Size * yb + parameters[roi_level].x_Size * xb;
            size_t f3Ind = parameters[roi_level].offset + parameters[roi_level].y_Size * yb + parameters[roi_level].x_Size * xa;
            size_t ind_out = OUTPUT_OFFSET + i * OUTPUT_Y_PITCH + j * OUTPUT_X_PITCH + kerNum * OUTPUT_BATCH_PITCH;

            for (int k = 0; k < OUTPUT_FEATURE_NUM; ++k) //transformation for every feature
            {
                INPUT0_TYPE f0 = feature_map_Ptr[roi_level][k * f_Size + f0Ind];
                INPUT0_TYPE f1 = feature_map_Ptr[roi_level][k * f_Size + f1Ind];
                INPUT0_TYPE f2 = feature_map_Ptr[roi_level][k * f_Size + f2Ind];
                INPUT0_TYPE f3 = feature_map_Ptr[roi_level][k * f_Size + f3Ind];

                INPUT0_TYPE f03 = f3 * b + f0 * a;
                INPUT0_TYPE f12 = f2 * b + f1 * a;
                INPUT0_TYPE f = f03 * c + f12 * d;

                output[k * OUTPUT_FEATURE_PITCH + ind_out] = f;
            }
            x += deltaX;
        }
        y += deltaY;
    }
}
