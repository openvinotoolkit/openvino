// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__constant static half log_2_e = (half)1.442695040888963; // log2(exp(1.0))

#define ALLOW_EARLY_RETURN 1

static void inline logistic_activate(__local const half *restrict src, __local half *restrict dst, int offset)
{
    half val    = src[offset];
    val         = 1.0h / (1.0h + exp2(val * -log_2_e));
    dst[offset] = val;
}

__kernel void region_chw(
    __global const half *restrict src_data,
    __global half *restrict dst_data,
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

    const int box_sz = W * H * (classes + coords + 1);
    event_t e1       = async_work_group_copy(local_src, src_data + get_group_id(1) * box_sz, box_sz, 0);
    wait_group_events(1, &e1);

    const int pixel_pos = get_local_id(0);
    const int stride    = W * H;

#if ALLOW_EARLY_RETURN
    if (pixel_pos < W * H)
#endif
    {
        __local const half *restrict src = local_src + pixel_pos;
        __local half *restrict dst       = local_dst + pixel_pos;

        logistic_activate(src, dst, 0 * stride);
        logistic_activate(src, dst, 1 * stride);

        //copy plane 2 and 3
        dst[2 * stride] = src[2 * stride];
        dst[3 * stride] = src[3 * stride];

        logistic_activate(src, dst, 4 * stride);

        src += (coords + 1) * stride;
        dst += (coords + 1) * stride;

        if (doSoftmax) {
            half max_val = src[0];
            #pragma unroll 4
            for (int c = 1; c < classes; c++) {
                max_val = max(max_val, src[c * stride]);
            }

            half expSum = 0.0h;
            #pragma unroll 4
            for (int c = 0; c < classes; c++) {
                const half e    = src[c * stride] - max_val;
                const half tmp  = exp2(e * log_2_e);
                dst[c * stride] = tmp;
                expSum += tmp;
            }

            const half recip = 1.h / expSum;
            int c            = 0;
            for (; c < (classes & ~0x3); c += 4) {
                const half t0 = dst[(c + 0) * stride];
                const half t1 = dst[(c + 1) * stride];
                const half t2 = dst[(c + 2) * stride];
                const half t3 = dst[(c + 3) * stride];

                const half e0 = t0 * recip;
                const half e1 = t1 * recip;
                const half e2 = t2 * recip;
                const half e3 = t3 * recip;

                dst[(c + 0) * stride] = e0;
                dst[(c + 1) * stride] = e1;
                dst[(c + 2) * stride] = e2;
                dst[(c + 3) * stride] = e3;
            }
            for (; c < classes; c++) {
                dst[c * stride] *= recip;
            }
        } else {
            #pragma unroll 4
            for (int c = 0; c < classes; c++) {
                logistic_activate(src, dst, c * stride);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy(dst_data + get_group_id(1) * box_sz, local_dst, box_sz, 0);
    wait_group_events(1, &e2);
}
