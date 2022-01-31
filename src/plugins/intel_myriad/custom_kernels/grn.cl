// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void grn(__global const half *restrict src_data, __global half *restrict dst_data, int C, float bias)
{
    __local half src[8 * 1024];
    __local half dst[8 * 1024];

    const size_t index = get_group_id(0) * get_local_size(0) + get_group_id(1) * get_local_size(1) * get_global_size(0);

    event_t e1 = async_work_group_copy_3D3D(
        src, // dst
        src_data + index, // src
        get_local_size(0), // num_elements_per_line,
        get_local_size(1), // num_lines,
        get_global_size(0) - get_local_size(0), // src_line_stride,
        0, // dst_line_stride,
        C, // num_planes,
        get_global_size(0) * (get_global_size(1) - get_local_size(1)), // src_plane_stride
        0, // dst_plane_stride
        0);
    wait_group_events(1, &e1);

    float variance = bias + 1e-9f;

    #pragma unroll 8
    for (int c = 0; c < C; c++) {
        float val = (float)src[c * get_local_size(1) * get_local_size(0)
                               + get_local_id(1) * get_local_size(0)
                               + get_local_id(0)];
        variance += val * val;
    }

    half hvariance = (half)(native_rsqrt((half)(variance / 16.f)) * 0.25f);

    #pragma unroll 8
    for (int c = 0; c < C; c++) {
        dst[c * get_local_size(1) * get_local_size(0)
            + get_local_id(1) * get_local_size(0)
            + get_local_id(0)] =
            src[c * get_local_size(1) * get_local_size(0)
                  + get_local_id(1) * get_local_size(0) + get_local_id(0)] * hvariance;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy_3D3D(
        dst_data + index, // src
        dst, // dst
        get_local_size(0), // num_elements_per_line,
        get_local_size(1), // num_lines,
        0, // src_line_stride,
        get_global_size(0) - get_local_size(0), // dst_line_stride,
        C, // num_planes,
        0, // src_plane_stride
        get_global_size(0) * (get_global_size(1) - get_local_size(1)), // dst_plane_stride
        0);
    wait_group_events(1, &e2);
}
