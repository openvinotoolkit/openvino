// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void reorg_chw(
    __global const half *restrict src,
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
        src + get_group_id(1) * W * stride
            + get_group_id(0) * W * stride * stride, // src
        W * stride, // num_elements_per_line,
        get_local_size(0), // num_lines,
        W * stride * (stride * get_num_groups(0) - 1), // src_line_stride,
        0, // dst_line_stride,
        0);
    wait_group_events(1, &e1);

    const int c        = get_local_id(0);
    const int stride_x = get_local_id(1);

    const int srcIdx = stride_x + c * W * stride;
    const int dstIdx = stride_x * W * get_local_size(0) + c * W;

    int x = 0;
    for (; x <= W - 8; x += 8) {
        half8 data = (half8){
            local_src[srcIdx + (x + 0) * stride],
            local_src[srcIdx + (x + 1) * stride],
            local_src[srcIdx + (x + 2) * stride],
            local_src[srcIdx + (x + 3) * stride],
            local_src[srcIdx + (x + 4) * stride],
            local_src[srcIdx + (x + 5) * stride],
            local_src[srcIdx + (x + 6) * stride],
            local_src[srcIdx + (x + 7) * stride]};

        *((__local half8 *)(&local_dst[dstIdx + x])) = data;
    }

    for (; x < W; x++) {
        local_dst[dstIdx + x] = local_src[srcIdx + x * stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy_2D2D(
        dst + get_group_id(0) * W
            + get_group_id(1) * W * stride * get_global_size(0), // dst
        local_dst, // src
        W, // num_elements_per_line
        get_local_size(0) * stride, // num_lines
        0, // src_line_stride
        W * (get_num_groups(0) - 1), // dst_line_stride
        0);
    wait_group_events(1, &e2);
}
