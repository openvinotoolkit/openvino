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

__kernel void grn_NCHW(__global const half* restrict src_data,
                       __global       half* restrict dst_data,
                       int C,
                       float bias)
{
    int x = get_global_id(0);
    int W = get_global_size(0);

    int y = get_global_id(1);
    int H = get_global_size(1);

    float variance = bias + 1e-9f;

    #pragma unroll 4
    for (int c = 0; c < C; c++)
    {
        float val = (float)src_data[c*H*W + y*W + x];
        variance += val * val;
    }

    variance = 1.f / native_sqrt(variance);

    #pragma unroll 4
    for (int c = 0; c < C; c++)
    {
        float val = (float)src_data[c*H*W + y*W + x];
        dst_data[c*H*W + y*W + x] = (half)(val * variance);
    }
}
