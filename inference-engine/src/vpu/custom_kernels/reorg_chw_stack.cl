// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define MAX_W 512

// kernel that uses private memory on stack
__kernel void reorg(__global const half* restrict src,
                    __global       half* restrict out,
                    int H,
                    int W,
                    int stride)
{
    int h = min((int)get_global_id(0), H-1);

    int c = get_global_id(1);
    int C = get_global_size(1);
    int C2 = C/(stride*stride);

    int offset = c / C2;

    int c2 = c - C2 * offset;

    int b = get_global_id(2);

    __private half tmp[MAX_W];

    int H2 = H*stride;
    int W2 = W*stride;

    for (int w = 0; w < W; ++w)
    {
        int h2 = h*stride + offset / stride;
        int w2 = w*stride + offset - stride * (offset / stride);

        tmp[w] = src[W2*H2*C2*b + W2*H2*c2 + W2*h2 + w2];
    }

    for (int w = 0; w < W; ++w)
    {
        out[W*H*C*b + W*H*c + W*h + w] = tmp[w];
    }
}
