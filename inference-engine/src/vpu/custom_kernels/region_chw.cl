// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define NUM_CLASSES 80

#define nlog_2_e ((half)(-1.442695040888963))

static void logistic_activate(__global const half* restrict src_data,
                              __global       half* restrict dst_data,
                              int offset)
{
    half val = src_data[offset];
    val = 1.f/(1.f + __builtin_shave_sau_exp2_f16_l_r(val*nlog_2_e));
    dst_data[offset] = val;
}

__kernel void region_ocl(__global const half* restrict src_data,
                         __global       half* restrict dst_data,
                         int W,
                         int H,
                         int classes,
                         int coords,
                         int num,
                         int maskSize,
                         int doSoftmax)
{
    int box_sz = H * W * (classes + coords + 1);
    int pixel_pos =  min((int)get_global_id(0), H*W);
    int box = get_global_id(1);

    //if (pixel_pos >= H*W) return;

    logistic_activate(src_data, dst_data, box * box_sz + pixel_pos + 0*H*W);
    logistic_activate(src_data, dst_data, box * box_sz + pixel_pos + 1*H*W);

    //copy plane 2 and 3
    dst_data[box * box_sz + pixel_pos + 2*H*W] = src_data[box * box_sz + pixel_pos + 2*H*W];
    dst_data[box * box_sz + pixel_pos + 3*H*W] = src_data[box * box_sz + pixel_pos + 3*H*W];

    logistic_activate(src_data, dst_data, box * box_sz + pixel_pos + 4*H*W);

    int data_offset =  box * box_sz + (coords + 1) * W * H;

    __private half data[NUM_CLASSES];

    if (doSoftmax) {
        half max_val = src_data[data_offset + 0*H*W + pixel_pos];
        for (int c = 0; c < classes; c++) {
            half tmp = src_data[data_offset + c*H*W + pixel_pos];
            data[c] = tmp;
            max_val = max( max_val, tmp);
        }

        half expSum = 0.0f;

        for (int c = 0; c < classes; c++) {
            half tmp = half_exp(data[c] - max_val);
            data[c] = tmp;
            expSum += tmp;
        }
        for (int c = 0; c < classes; c++) {
            data[c] = data[c] / expSum;
        }

        for (int c = 0; c < classes; c++) {
            dst_data[data_offset + c*H*W + pixel_pos + 0] = data[c];
        }
    }
    else {
        for (int i = 0;  i < classes; i++) {
            logistic_activate(src_data, dst_data, box * box_sz + pixel_pos + (5 + i)*H*W);
        }
    }
}
