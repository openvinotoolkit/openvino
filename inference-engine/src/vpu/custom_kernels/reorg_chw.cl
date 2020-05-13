// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define USE_MANUAL_DMA

#if defined (USE_MANUAL_DMA)

__kernel void __dma_preload_reorg_chw(__global half const *restrict src,
                                      __global half       *restrict dst,
                                      int W,
                                      int H,
                                      int C,
                                      int stride,
                                      __local half        *restrict local_src,
                                      __local half        *restrict local_dst
                                      )
{
    const int stride_y = get_group_id(1);

    const int srcIdx = stride_y*W*stride + W*stride*stride*get_group_id(0);

    WorkGroupDmaCreateStrideTransaction(
        src + srcIdx, // src
        local_src, // dst
        W * stride * sizeof(half), // src width
        W * stride * sizeof(half), // dst width
        W * stride * stride * get_num_groups(0) * sizeof(half), // src stride
        W * stride * sizeof(half),  // dst stride
        W * stride * get_local_size(0) * sizeof(half), //total size
        0);
}

__kernel void __dma_postwrite_reorg_chw(__global half const *restrict src,
                                        __global half       *restrict dst,
                                        int W,
                                        int H,
                                        int C,
                                        int stride,
                                        __local half       *restrict local_src,
                                        __local half const *restrict local_dst
                                        )
{
    const int stride_y = get_group_id(1);

    const int dstIdx = stride_y*W*stride*get_global_size(0) + get_group_id(0)*W;

    WorkGroupDmaCreateStrideTransaction(
        local_dst, // src
        dst + dstIdx, // dst
        W * sizeof(half), // src width
        W * sizeof(half), // dst width
        W * sizeof(half), // src stride
        W * get_num_groups(0) * sizeof(half),  // dst stride
        get_local_size(0) * W * stride * sizeof(half), //total size
        0);
}

__kernel void reorg_chw(__global half const *restrict src,
                        __global half       *restrict dst,
                        int W,
                        int H,
                        int C,
                        int stride,
                        __local half       *restrict local_src,
                        __local half       *restrict local_dst
                        )
{
    const int c = get_local_id(0);
    const int stride_x = get_local_id(1);

    const int srcIdx = stride_x + c*W*stride;
    const int dstIdx = stride_x*W*get_local_size(0) + c*W;

    int x = 0;
    for (; x <= W - 8; x += 8) {
         half8 data = (half8) {
             local_src[srcIdx + (x + 0)*stride], local_src[srcIdx + (x + 1)*stride],
             local_src[srcIdx + (x + 2)*stride], local_src[srcIdx + (x + 3)*stride],
             local_src[srcIdx + (x + 4)*stride], local_src[srcIdx + (x + 5)*stride],
             local_src[srcIdx + (x + 6)*stride], local_src[srcIdx + (x + 7)*stride]
         };

         *((__local half8*)(&local_dst[dstIdx + x])) = data;
    }

    for (; x < W; x++) {
        local_dst[dstIdx + x] = local_src[srcIdx + x*stride];
    }
}

#else

__kernel void reorg_chw(__global half const *restrict src,
                        __global half       *restrict dst,
                        int W,
                        int H,
                        int C,
                        int stride,
                        __local half const *restrict _0,
                        __local half       *restrict _1
                        )
{
    const int stride_x = get_local_id(1);
    const int stride_y = get_group_id(1);
    const int N = get_global_size(0);
    const int c = get_local_id(0)*get_num_groups(0) + get_group_id(0);

    const int srcIdx = c*W*stride*stride + stride_x + stride_y*W*stride;
    const int dstIdx = c*W + stride_x*W*N + stride_y*W*N*stride;

    #pragma unroll 8
    for (int x = 0; x < W; x++) {
        dst[dstIdx + x] = src[srcIdx + x*stride];
    }
}

#endif

