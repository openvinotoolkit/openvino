// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__constant static half log_2_e = (half)1.442695040888963; // log2(exp(1.0))

#define ALLOW_EARLY_RETURN 1

static void inline logistic_activate_hwc(
    __local const half *restrict src,
    __local half *restrict dst,
    int offset,
    int stride)
{
    half val             = src[offset];
    val                  = 1.0h / (1.0h + exp2(val * -log_2_e));
    dst[offset * stride] = val;
}

__kernel void region_hwc(
    __global const half *restrict src,
    __global half *restrict dst,
    int W,
    int H,
    int classes,
    int coords,
    int num,
    int maskSize,
    int doSoftmax)
{
    __local half local_src[13 * 13 * (4 + 1 + 80)];
    __local half local_dst[13 * 13 * (4 + 1 + 80)];

    const int pixel_pos = get_local_id(0);

    const int local_C = classes + coords + 1;
    const int c       = get_group_id(1) * local_C;
    const int h       = get_group_id(0);

    num         = (doSoftmax != 0) * num + (doSoftmax == 0) * maskSize;
    const int C = local_C * num;

    event_t e1 = async_work_group_copy_2D2D(
        local_src, // dst
        src + h * W * C + c, // src
        local_C, // num_elements_per_line,
        H * W, // num_lines,
        C - local_C, // src_line_stride,
        0, // dst_line_stride,
        0);

    wait_group_events(1, &e1);

#if ALLOW_EARLY_RETURN
    if (pixel_pos < W * H)
#endif
    {
        const int w = pixel_pos % W;
        const int h = pixel_pos / W;

        __local const half *restrict src = local_src + h * W * local_C + w * local_C;
        __local half *restrict dst       = local_dst + h * W + w;

        const int stride = H * W;
        logistic_activate_hwc(src, dst, 0, stride);
        logistic_activate_hwc(src, dst, 1, stride);

        //copy plane 2 and 3
        dst[2 * stride] = src[2];
        dst[3 * stride] = src[3];

        logistic_activate_hwc(src, dst, 4, stride);

        src += coords + 1;
        dst += (coords + 1) * stride;

        if (doSoftmax) {
            half max_val = src[0];
            #pragma unroll 4
            for (int c = 1; c < classes; c++) {
                max_val = max(max_val, src[c]);
            }

            half expSum = 0.0h;
            #pragma unroll 4
            for (int c = 0; c < classes; c++) {
                const half e    = src[c] - max_val;
                const half tmp  = exp2(e * log_2_e);
                dst[c * stride] = tmp;
                expSum += tmp;
            }

            const half invExpSum = 1.0h / expSum;
            #pragma unroll 4
            for (int c = 0; c < classes; c++) {
                dst[c * stride] *= invExpSum;
            }
        } else {
            #pragma unroll 4
            for (int c = 0; c < classes; c++) {
                logistic_activate_hwc(src, dst, c, stride);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int box_sz = W * H * (classes + coords + 1);
    event_t e2       = async_work_group_copy(dst + get_group_id(1) * box_sz, local_dst, box_sz, 0);
    wait_group_events(1, &e2);
}
