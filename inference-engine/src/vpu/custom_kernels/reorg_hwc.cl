// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void __dma_preload_reorg_hwc(__global half const *restrict src,
                                        __global half       *restrict _0,
                                        int W,
                                        int H,
                                        int C,
                                        int stride,
                                        __local half       *restrict local_src,
                                        __local half       *restrict _1
                                        )
{
    const int stride_x = get_group_id(1);

    WorkGroupDmaCreateStrideTransaction(
        src + get_group_id(0) * stride + stride_x * C, // src
        local_src, // dst
        stride * sizeof(half), // src_width,
        stride * sizeof(half), // dst_width,
        C * stride * sizeof(half), // src_stride,
        stride * sizeof(half), // dst_stride,
        H * W * sizeof(half), // size
        0);
}

__kernel void __dma_postwrite_reorg_hwc(__global half const *restrict _0,
                                        __global half       *restrict dst,
                                        int W,
                                        int H,
                                        int C,
                                        int stride,
                                        __local half       *restrict _1,
                                        __local half       *restrict local_dst
                                        )
{
    const int stride_x = get_group_id(1);

    WorkGroupDmaCreateStrideTransaction(
        local_dst, // src
        dst + stride_x * C + get_group_id(0) * stride, // dst
        stride * sizeof(half), // src_width,
        stride * sizeof(half), // dst_width,
        stride * sizeof(half), // src_stride,
        C * stride * sizeof(half), // dst_stride,
        W * H * sizeof(half), // size
        0);
}

__kernel void reorg_hwc(__global half const *restrict src,
                        __global half       *restrict dst,
                        int W,
                        int H,
                        int C,
                        int stride,
                        __local half        *restrict local_src,
                        __local half        *restrict local_dst
                        )
{
    const int stride_y = get_local_id(1);
    const int blocks = get_local_size(0);
    const int b = get_local_id(0);

    const int OC = stride * stride;
    const int OH = H / stride;
    const int OW = W / stride;
    const int IC = stride;
    const int IH = H;
    const int IW = W / stride;

    for (int block_h = 0; block_h < stride; block_h++) {
        const int src_line = b * stride * stride + stride_y * stride + block_h;
        const int c = src_line / IH;
        const int h = src_line % IH;

        const int dst_line = b * stride + stride_y * blocks * stride + block_h;
        const int oc = dst_line / OH;
        const int oh = dst_line % OH;

        for (int w = 0; w < W / stride; w++) {
            local_dst[oh*OW*OC + w*OC + oc] = local_src[h*IW*IC + w*IC + c];
        }
    }
}

__kernel void reorg_hwc_naive(__global half const *restrict src,
                              __global half       *restrict dst,
                              int W,
                              int H,
                              int C,
                              int stride,
                              __local half        *restrict local_src,
                              __local half        *restrict local_dst
                              )
{
    const int out_c = C / (stride * stride);
    const int oc = C * (stride * stride);
    const int oh = H / stride;
    const int ow = W / stride;

    const int c = get_global_id(0);

    for (int h = 0; h < H; ++h)
    {
        int in_index = W * (h + H*c) + (0);
        int new_z = in_index / (oh*ow);
        int new_y = (in_index %(oh*ow)) / ow;
        int new_x = (in_index %(oh*ow)) % ow;
        int new_index = new_z + new_x * oc + new_y * oc * ow;

        in_index++;

        int c2 = c % out_c;
        int offset = c / out_c;
        int w2 = 0 * stride + offset % stride;
        int h2 = h * stride + offset / stride;
        int out_index = w2 + W * stride * (h2 + H * stride * c2);

        #pragma unroll 2
        for(int i = 0; i < W; ++i, out_index+=stride, in_index++)
        {
            // repacking coordinates
            int k0 = out_index / (H*W);
            int j0 = (out_index % (H*W)) / W;
            int i0 = (out_index % (H*W)) % W;
            int out_index_repack = k0 + C * i0 + C * W * j0;

            dst[new_index] = src[out_index_repack];

            int new_z = in_index / (oh*ow);
            int new_y = (in_index %(oh*ow)) / ow;
            int new_x = (in_index %(oh*ow)) % ow;
            new_index = new_z + new_x * oc + new_y * oc * ow;
        }
    }
}
