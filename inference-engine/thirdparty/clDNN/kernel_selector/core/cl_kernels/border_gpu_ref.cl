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
#if INPUT0_DIMS >= 5
    const uint blt_sz = LT_SIZES_SIZE_Z;
#else
    const uint blt_sz = 0;
#endif
#if INPUT0_DIMS == 6
    const uint blt_sw = LT_SIZES_SIZE_W;
#else
    const uint blt_sw = 0;
#endif
    const uint blt_sf = LT_SIZES_FEATURE_NUM;
    const uint blt_sb = LT_SIZES_BATCH_NUM;

    const uint brb_sx = RB_SIZES_SIZE_X;
    const uint brb_sy = RB_SIZES_SIZE_Y;
#if INPUT0_DIMS >= 5
    const uint brb_sz = RB_SIZES_SIZE_Z;
#else
    const uint brb_sz = 0;
#endif
#if INPUT0_DIMS == 6
    const uint brb_sw = RB_SIZES_SIZE_W;
#else
    const uint brb_sw = 0;
#endif
    const uint brb_sf = RB_SIZES_FEATURE_NUM;
    const uint brb_sb = RB_SIZES_BATCH_NUM;

    // Input sizes:
    const uint in_sx = INPUT0_SIZE_X;
    const uint in_sy = INPUT0_SIZE_Y;
    const uint in_sz = INPUT0_SIZE_Z;
    const uint in_sw = INPUT0_SIZE_W;
    const uint in_sf = INPUT0_FEATURE_NUM;
    const uint in_sb = INPUT0_BATCH_NUM;

    // Input limits (exclusive; tested on output position):
    const uint in_lx = in_sx + blt_sx;
    const uint in_ly = in_sy + blt_sy;
    const uint in_lz = in_sz + blt_sz;
    const uint in_lw = in_sw + blt_sw;
    const uint in_lf = in_sf + blt_sf;
    const uint in_lb = in_sb + blt_sb;  

    const uint out_xz  = (uint) get_global_id(0);
    const uint out_yw  = (uint) get_global_id(1);
    const uint out_fb = (uint) get_global_id(2);

    const uint out_f  = out_fb % OUTPUT_FEATURE_NUM;
    const uint out_b  = out_fb / OUTPUT_FEATURE_NUM;

    const uint out_x  = out_xz % OUTPUT_SIZE_X;
    const uint out_z  = out_xz / OUTPUT_SIZE_X;

    const uint out_y  = out_yw % OUTPUT_SIZE_Y;
    const uint out_w  = out_yw / OUTPUT_SIZE_Y;

#ifdef BORDER_TYPE_CONSTANT
    UNIT_TYPE in_val = TO_UNIT_TYPE(BORDER_VALUE);

    if (out_x >= blt_sx & out_x < in_lx &
        out_y >= blt_sy & out_y < in_ly &
        out_z >= blt_sz & out_z < in_lz &
        out_w >= blt_sw & out_w < in_lw &
        out_f >= blt_sf & out_f < in_lf &
        out_b >= blt_sb & out_b < in_lb)
    {
        const uint in_x = out_x - blt_sx;
        const uint in_y = out_y - blt_sy;
        const uint in_z = out_z - blt_sz;
        const uint in_w = out_w - blt_sw;
        const uint in_f = out_f - blt_sf;
        const uint in_b = out_b - blt_sb;

        const uint in_pos = GET_DATA_INDEX_6D(INPUT0, in_b, in_f, in_w, in_z, in_y, in_x);
        in_val = input[in_pos];
    }
#elif defined BORDER_TYPE_EDGE
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? 0 : in_sx - 1);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? 0 : in_sy - 1);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? 0 : in_sz - 1);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? 0 : in_sw - 1);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? 0 : in_sf - 1);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? 0 : in_sb - 1);

    const uint in_pos = GET_DATA_INDEX_6D(INPUT0, in_b, in_f, in_w, in_z, in_y, in_x);
    UNIT_TYPE in_val = input[in_pos];
#elif defined BORDER_TYPE_MIRROR
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? blt_sx - 1 - out_x : in_sx + in_lx - 1 - out_x);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? blt_sy - 1 - out_y : in_sy + in_ly - 1 - out_y);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? blt_sz - 1 - out_z : in_sz + in_lz - 1 - out_z);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? blt_sw - 1 - out_w : in_sw + in_lw - 1 - out_w);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? blt_sf - 1 - out_f : in_sf + in_lf - 1 - out_f);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? blt_sb - 1 - out_b : in_sb + in_lb - 1 - out_b);

    const uint in_pos = GET_DATA_INDEX_6D(INPUT0, in_b, in_f, in_w, in_z, in_y, in_x);
    UNIT_TYPE in_val = input[in_pos];
#elif defined BORDER_TYPE_MIRROR_101
    const uint in_x = (out_x >= blt_sx & out_x < in_lx) ? out_x - blt_sx : (out_x < blt_sx ? blt_sx - out_x : in_sx + in_lx - 2 - out_x);
    const uint in_y = (out_y >= blt_sy & out_y < in_ly) ? out_y - blt_sy : (out_y < blt_sy ? blt_sy - out_y : in_sy + in_ly - 2 - out_y);
    const uint in_z = (out_z >= blt_sz & out_z < in_lz) ? out_z - blt_sz : (out_z < blt_sz ? blt_sz - out_z : in_sz + in_lz - 2 - out_z);
    const uint in_w = (out_w >= blt_sw & out_w < in_lw) ? out_w - blt_sw : (out_w < blt_sw ? blt_sw - out_w : in_sw + in_lw - 2 - out_w);
    const uint in_f = (out_f >= blt_sf & out_f < in_lf) ? out_f - blt_sf : (out_f < blt_sf ? blt_sf - out_f : in_sf + in_lf - 2 - out_f);
    const uint in_b = (out_b >= blt_sb & out_b < in_lb) ? out_b - blt_sb : (out_b < blt_sb ? blt_sb - out_b : in_sb + in_lb - 2 - out_b);

    const uint in_pos = GET_DATA_INDEX_6D(INPUT0, in_b, in_f, in_w, in_z, in_y, in_x);
    UNIT_TYPE in_val = input[in_pos];
#else
    #error Unsupported border type.
#endif

    const uint out_pos = GET_DATA_INDEX_6D(OUTPUT, out_b, out_f, out_w, out_z, out_y, out_x);
    output[out_pos] = in_val;
}
