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


KERNEL(border_gpu_ref)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output)
{
    // [CONSTEXPR]
    // Border sizes (left-top set and right-bottom set):
    const uint blt_sx = LT_SIZES_SIZE_X;
    const uint blt_sy = LT_SIZES_SIZE_Y;
    const uint blt_sf = LT_SIZES_FEATURE_NUM;
    const uint blt_sb = LT_SIZES_BATCH_NUM;

    const uint brb_sx = RB_SIZES_SIZE_X;
    const uint brb_sy = RB_SIZES_SIZE_Y;
    const uint brb_sf = RB_SIZES_FEATURE_NUM;
    const uint brb_sb = RB_SIZES_BATCH_NUM;

    // Input sizes:
    const uint in_sx = INPUT0_SIZE_X;
    const uint in_sy = INPUT0_SIZE_Y;
    const uint in_sf = INPUT0_FEATURE_NUM;
    const uint in_sb = INPUT0_BATCH_NUM;

    // Input limits (exclusive; tested on output position):
    const uint in_lx = in_sx + blt_sx;
    const uint in_ly = in_sy + blt_sy;
    const uint in_lf = in_sf + blt_sf;
    const uint in_lb = in_sb + blt_sb;


    const uint out_x  = (uint) get_global_id(0);
    const uint out_y  = (uint) get_global_id(1);
    const uint out_fb = (uint) get_global_id(2);

    const uint out_f  = out_fb % OUTPUT_FEATURE_NUM;
    const uint out_b  = out_fb / OUTPUT_FEATURE_NUM;

#ifdef BORDER_TYPE_ZERO
    UNIT_TYPE in_val = UNIT_VAL_ZERO;
    if (out_x >= blt_sx & out_x < in_lx &
        out_y >= blt_sy & out_y < in_ly &
        out_f >= blt_sf & out_f < in_lf &
        out_b >= blt_sb & out_b < in_lb)
    {
        const uint in_x = out_x - blt_sx;
        const uint in_y = out_y - blt_sy;
        const uint in_f = out_f - blt_sf;
        const uint in_b = out_b - blt_sb;

        const uint in_pos = GET_DATA_INDEX(INPUT0, in_b, in_f, in_y, in_x);
        in_val = input[in_pos];
    }
#elif defined BORDER_TYPE_MIRROR
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? blt_sx - 1 - out_x : in_sx + in_lx - 1 - out_x);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? blt_sy - 1 - out_y : in_sy + in_ly - 1 - out_y);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? blt_sf - 1 - out_f : in_sf + in_lf - 1 - out_f);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? blt_sb - 1 - out_b : in_sb + in_lb - 1 - out_b);

    const uint in_pos = GET_DATA_INDEX(INPUT0, in_b, in_f, in_y, in_x);
    UNIT_TYPE in_val = input[in_pos];
#elif defined BORDER_TYPE_MIRROR_101
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? blt_sx - out_x : in_sx + in_lx - 2 - out_x);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? blt_sy - out_y : in_sy + in_ly - 2 - out_y);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? blt_sf - out_f : in_sf + in_lf - 2 - out_f);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? blt_sb - out_b : in_sb + in_lb - 2 - out_b);

    const uint in_pos = GET_DATA_INDEX(INPUT0, in_b, in_f, in_y, in_x);
    UNIT_TYPE in_val = input[in_pos];
#else
    #error Unsupported border type.
#endif

    const uint out_pos = GET_DATA_INDEX(OUTPUT, out_b, out_f, out_y, out_x);
    output[out_pos] = in_val;
}