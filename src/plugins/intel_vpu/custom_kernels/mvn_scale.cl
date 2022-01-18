// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

// Set to 1 only if output is zerroed before kernel execution
#define USE_ATOMICS 0

__attribute__((reqd_work_group_size(1, 1, 1))) __kernel void mvn_scale(
    const __global half *restrict src,
    __global float *restrict mean_part,
    __global float *restrict power_mean,
    __global half *restrict dst,
    int W,
    int H1,
    int across_channels,
    int normalize_variance,
    int nparts)
{
    __local half src_line[4 * 1024];
    __local half dst_line[4 * 1024];

    int c = get_group_id(2);
    int C = get_global_size(2);

    int h = get_group_id(1);
    int H = get_global_size(1);

    event_t e1 = async_work_group_copy(src_line, src + c * H * W + h * W, W, 0);
    wait_group_events(1, &e1);

    int idx     = (across_channels == 0) ? nparts * c : 0;
    float scale = (across_channels == 0) ? H * W : H * W * C;

#if USE_ATOMICS
    float mean     = mean_part[idx];
    float variance = power_mean[idx];
#else

    int total      = (across_channels == 0) ? nparts : nparts * C;
    float mean     = 0.f;
    float variance = 0.f;

    for (int i = 0; i < total; i++) {
        mean += mean_part[idx + i];
        variance += power_mean[idx + i];
    }
#endif

    mean     = mean / scale;
    variance = variance / scale;
    variance = variance - mean * mean;
    variance = native_sqrt(variance) + 1e-9f;

    half hmean     = mean;
    half hvariance = (normalize_variance == 0) ? 1.f : (1.f / variance);

    for (size_t w = 0; w < W; w++) {
        dst_line[w] = (src_line[w] - hmean) * hvariance;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy(dst + c * H * W + h * W, dst_line, W, 0);
    wait_group_events(1, &e2);
}
