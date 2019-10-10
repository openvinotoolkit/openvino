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

__kernel void ShuffleChannel(__global const half* restrict src_data,
                             __global       half* restrict dst_data,
                             int C,
                             int H,
                             int W,
                             int G)
{
    int c = get_global_id(0);
    if (c >= C) return;
    int CX = C / G;
    int CY = G;
    int cy = c % G;
    int cx = c / G;

    __global const half8* src_line = ((__global const half8*)(src_data + cy*CX*H*W + cx*H*W));
    __global       half8* dst_line = ((__global       half8*)(dst_data + cx*CY*H*W + cy*H*W));

    for (int i = 0; i < W*H/8; i++)
    {
        dst_line[i] = src_line[i];
    }

    for (int i = W*H/8*8; i < W*H; i++)
    {
        dst_data[cx*CY*H*W + cy*H*W + i] = src_data[cy*CX*H*W + cx*H*W + i];
    }
}
