// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void ShuffleChannel(
    __global const half *restrict src_data,
    __global half *restrict dst_data,
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

    __global const half8 *src_line =
        ((__global const half8 *)(src_data + cy * CX * H * W + cx * H * W));
    __global half8 *dst_line = ((__global half8 *)(dst_data + cx * CY * H * W + cy * H * W));

    for (int i = 0; i < W * H / 8; i++) {
        dst_line[i] = src_line[i];
    }

    for (int i = W * H / 8 * 8; i < W * H; i++) {
        dst_data[cx * CY * H * W + cy * H * W + i] = src_data[cy * CX * H * W + cx * H * W + i];
    }
}
