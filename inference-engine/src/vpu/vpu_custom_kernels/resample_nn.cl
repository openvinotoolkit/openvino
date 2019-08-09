// Copyright (C) 2019 Intel Corporation
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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void resample_nearest(__global const half* restrict src,
                             __global       half* restrict dst,
                             int iw,
                             int ih,
                             float fx,
                             float fy,
                             int ow,
                             int oh,
                             int channels)
{
    int oy = min((int)get_global_id(0), oh-1);
    int c = get_global_id(1);
    int b = get_global_id(2);

    __global const half* start_src = src + b * iw * ih * channels + iw * ih * c;
    __global       half* start_dst = dst + b * ow * oh * channels + ow * oh * c;

    for (int ox = 0; ox < ow; ox++)
    {
        float ix_r0 = ox*fx + fx / 2.0f - 0.5f;
        float iy_r0 = oy*fy + fy / 2.0f - 0.5f;
        int ix_r1 = (int)(round(ix_r0));
        int iy_r1 = (int)(round(iy_r0));
        start_dst[oy * ow + ox] = start_src[iy_r1 * iw + ix_r1];
    }
}
