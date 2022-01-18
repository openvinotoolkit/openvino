// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void reorg_hwc(
    __global half const *restrict src,
    __global half *restrict dst,
    int W,
    int H,
    int C,
    int stride)
{
    __local half local_src[8 * 1024];
    __local half local_dst[8 * 1024];

    event_t e1 = async_work_group_copy_2D2D(
        local_src, // dst
        src + get_group_id(0) * stride + get_group_id(1) * C, // src
        stride, // num_elements_per_line
        H * W / stride, // num_lines
        (C - 1) * stride, // src_line_stride
        0, // dst_line_stride
        0);
    wait_group_events(1, &e1);

    const int stride_y = get_local_id(1);
    const int blocks   = get_local_size(0);
    const int b        = get_local_id(0);

    const int OC = stride * stride;
    const int OH = H / stride;
    const int OW = W / stride;
    const int IC = stride;
    const int IH = H;
    const int IW = W / stride;

    for (int block_h = 0; block_h < stride; block_h++) {
        const int src_line = b * stride * stride + stride_y * stride + block_h;
        const int c        = src_line / IH;
        const int h        = src_line % IH;

        const int dst_line = b * stride + stride_y * blocks * stride + block_h;
        const int oc       = dst_line / OH;
        const int oh       = dst_line % OH;

        for (int w = 0; w < W / stride; w++) {
            local_dst[oh * OW * OC + w * OC + oc] = local_src[h * IW * IC + w * IC + c];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy_2D2D(
        dst + get_group_id(1) * C + get_group_id(0) * stride, // dst
        local_dst, // src
        stride, // num_elements_per_line
        W * H / stride, // num_lines
        0, // src_line_stride
        C * stride - stride, // dst_line_stride
        0);
    wait_group_events(1, &e2);
}
