// Copyright (C) 2018-2020 Intel Corporation
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

#define NUM_CLASSES 80

static void logistic_activate(__global const half* restrict src_data,
                              __global       half* restrict dst_data,
                              int offset)
{
    half val = src_data[offset];
    val = 1.0f/(1.0f + native_exp(-val));
    dst_data[offset] = val;
}

__kernel void region_ocl(__global const half* restrict src_data,
                         __global       half* restrict dst_data,
                         int W,
                         int H,
                         int classes,
                         int coords)
{
    int box_sz = H * W * (classes + coords + 1);
    int pixel_pos = min((int)get_global_id(0), ((H*W) - 1));
    int box = get_global_id(1);

    logistic_activate(src_data, dst_data, box * box_sz + pixel_pos + 0*H*W);
    logistic_activate(src_data, dst_data, box * box_sz + pixel_pos + 1*H*W);

    //copy plane 2 and 3
    dst_data[box * box_sz + pixel_pos + 2*H*W] = src_data[box * box_sz + pixel_pos + 2*H*W];
    dst_data[box * box_sz + pixel_pos + 3*H*W] = src_data[box * box_sz + pixel_pos + 3*H*W];

    logistic_activate(src_data, dst_data, box * box_sz + pixel_pos + 4*H*W);

    int data_offset =  box * box_sz + (coords + 1) * W * H;

    for (int i = 0;  i < classes; i++) {
        logistic_activate(src_data, dst_data, box * box_sz + pixel_pos + (5 + i)*H*W);
    }
}
