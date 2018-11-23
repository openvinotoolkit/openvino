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


KERNEL(broadcast_gpu_ref)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output)
{
    // [CONSTEXPR]
    // Input sizes:
    const uint in_sx = INPUT0_SIZE_X;
    const uint in_sy = INPUT0_SIZE_Y;
    const uint in_sf = INPUT0_FEATURE_NUM;
    const uint in_sb = INPUT0_BATCH_NUM;


    const uint out_x  = (uint) get_global_id(0);
    const uint out_y  = (uint) get_global_id(1);
    const uint out_fb = (uint) get_global_id(2);

    const uint out_f  = out_fb % OUTPUT_FEATURE_NUM;
    const uint out_b  = out_fb / OUTPUT_FEATURE_NUM;


    const uint in_x = out_x % in_sx;
    const uint in_y = out_y % in_sy;
    const uint in_f = out_f % in_sf;
    const uint in_b = out_b % in_sb;

    const uint in_pos  = GET_DATA_INDEX(INPUT0, in_b,  in_f,  in_y,  in_x);
    const uint out_pos = GET_DATA_INDEX(OUTPUT, out_b, out_f, out_y, out_x);


    output[out_pos] = input[in_pos];
}