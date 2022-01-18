// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void cvtu8f16(__global const uchar *restrict src, __global half *restrict dst, float scale, float bias)
{
    __local uchar local_src[8 * 1024];
    __local half local_dst[8 * 1024];

    event_t e1 = async_work_group_copy_3D3D(
        local_src, // dst
        src + get_group_id(0) * get_local_size(0) + get_group_id(1) * get_local_size(1) * get_global_size(0)
            + get_group_id(2) * get_local_size(2) * get_global_size(0) * get_global_size(1), // src
        get_local_size(0), // num_elements_per_line
        get_local_size(0) * get_local_size(1) / (get_local_size(0)), // num_lines
        get_global_size(0) - get_local_size(0), // src_line_stride
        0, // dst_line_stride
        get_local_size(2), // num planes
        get_global_size(0) * (get_global_size(1) - get_local_size(1)), // src plane stride
        0, // dst plane stride
        0);
    wait_group_events(1, &e1);

    size_t idx = get_local_id(0)
               + get_local_id(1) * get_local_size(0)
               + get_local_id(2) * get_local_size(0) * get_local_size(1);

    local_dst[idx] = convert_half(local_src[idx]) * (half)scale + (half)bias;

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy_3D3D(
        dst + get_group_id(0) * get_local_size(0) + get_group_id(1) * get_local_size(1) * get_global_size(0)
            + get_group_id(2) * get_local_size(2) * get_global_size(0) * get_global_size(1), // dst
        local_dst, // src
        get_local_size(0), // num_elements_per_line
        get_local_size(1), // num_lines
        0, // src_line_stride
        get_global_size(0) - get_local_size(0), // dst_line_stride
        get_local_size(2), // num_planes
        0, // src_plane_stride
        get_global_size(0) * (get_global_size(1) - get_local_size(1)), // dst_plane_stride
        0);
    wait_group_events(1, &e2);
}
