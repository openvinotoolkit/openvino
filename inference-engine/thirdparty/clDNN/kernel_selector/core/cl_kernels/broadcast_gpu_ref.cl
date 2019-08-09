// Copyright (c) 2018-2019 Intel Corporation
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


KERNEL(broadcast_gpu_ref)(
    const __global INPUT0_TYPE* input,
    __global INPUT0_TYPE* output)
{
    // [CONSTEXPR]
    // Input sizes:
#ifdef OUTPUT_LAYOUT_BFZYX
    uint8 input_indices;
#else
    uint4 input_indices;
#endif
    input_indices[0] = INPUT0_BATCH_NUM;
    input_indices[1] = INPUT0_FEATURE_NUM;
#ifdef OUTPUT_LAYOUT_BFZYX
    input_indices[2] = INPUT0_SIZE_Z;
    input_indices[3] = INPUT0_SIZE_Y;
    input_indices[4] = INPUT0_SIZE_X;
#else
    input_indices[2] = INPUT0_SIZE_Y;
    input_indices[3] = INPUT0_SIZE_X;
#endif
#ifdef OUTPUT_LAYOUT_BFZYX
    const uint in_sx = input_indices[BROADCAST_ORDER[4]];
    const uint in_sy = input_indices[BROADCAST_ORDER[3]];
    const uint in_sz = input_indices[BROADCAST_ORDER[2]];
#else
    const uint in_sx = input_indices[BROADCAST_ORDER[3]];
    const uint in_sy = input_indices[BROADCAST_ORDER[2]];
#endif
    const uint in_sf = input_indices[BROADCAST_ORDER[1]];
    const uint in_sb = input_indices[BROADCAST_ORDER[0]];

    const uint out_x  = (uint) get_global_id(0);
#ifdef OUTPUT_LAYOUT_BFZYX
    const uint out_zy = (uint) get_global_id(1);
    const uint out_y  = out_zy % OUTPUT_SIZE_Y;
    const uint out_z  = out_zy / OUTPUT_SIZE_Y;
#else
    const uint out_y = (uint) get_global_id(1);
#endif

    const uint out_fb = (uint) get_global_id(2);

    const uint out_f  = out_fb % OUTPUT_FEATURE_NUM;
    const uint out_b  = out_fb / OUTPUT_FEATURE_NUM;

    const uint in_x = out_x % in_sx;
    const uint in_y = out_y % in_sy;
#ifdef OUTPUT_LAYOUT_BFZYX
    const uint in_z = out_z % in_sz;
#endif
    const uint in_f = out_f % in_sf;
    const uint in_b = out_b % in_sb;
#ifdef OUTPUT_LAYOUT_BFZYX
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_z + in_sz * (in_f + in_sf * in_b)));
    const uint out_pos = GET_DATA_INDEX_5D(OUTPUT, out_b, out_f, out_z, out_y, out_x);
#else
    const uint in_pos =  INPUT0_OFFSET + in_x + in_sx * (in_y + in_sy * (in_f + in_sf * in_b));
    const uint out_pos = GET_DATA_INDEX(OUTPUT, out_b, out_f, out_y, out_x);
#endif
    output[out_pos] = input[in_pos];
}
