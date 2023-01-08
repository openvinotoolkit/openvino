// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void binarization(
    const __global half *__restrict src_data,
    const __global half *__restrict input_low_high,
    const __global half *__restrict dst_data,
    int switch_out,
    int input_low_high_size,
    int W,
    int H)
{
    __local half local_src[15 * 1024];
    __local half local_dst[15 * 1024];

    event_t e1 = async_work_group_copy(local_src, src_data + get_group_id(2) * W * H, W * H, 0);
    wait_group_events(1, &e1);

    int c = get_global_id(2);
    int C = get_global_size(2);

    half dst_low  = switch_out ? 1.h : -1.h;
    half dst_high = switch_out ? -1.h : 1.h;

    half s_ilow_ihigh = input_low_high_size == 1 ? input_low_high[0] : input_low_high[c];

    for (int h = 0; h < H; h++) {

        __local const half *__restrict addr_src = local_src + h * W;
        __local half *__restrict addr_dst       = local_dst + h * W;

#if 1
        for (int w = 0; w < W / 8; w++) {

            half8 h_src_val8 = (*((__local half8 *)addr_src + w));

            short8 cond1;
            cond1.s0 = (h_src_val8.s0 <= s_ilow_ihigh);
            cond1.s1 = (h_src_val8.s1 <= s_ilow_ihigh);
            cond1.s2 = (h_src_val8.s2 <= s_ilow_ihigh);
            cond1.s3 = (h_src_val8.s3 <= s_ilow_ihigh);
            cond1.s4 = (h_src_val8.s4 <= s_ilow_ihigh);
            cond1.s5 = (h_src_val8.s5 <= s_ilow_ihigh);
            cond1.s6 = (h_src_val8.s6 <= s_ilow_ihigh);
            cond1.s7 = (h_src_val8.s7 <= s_ilow_ihigh);

            cond1 = ~(cond1 - (short8)1);

            short8 res = cond1 & as_short8((half8)dst_low) | ~cond1 & as_short8((half8)dst_high);

            *((__local half8 *)addr_dst + w) = as_half8(res);
        }
#endif
        for (int w = W & (~0x7); w < W; w++) {
            addr_dst[w] = (addr_src[w] <= s_ilow_ihigh) ? dst_low : dst_high;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy(dst_data + get_group_id(2) * W * H, local_dst, W * H, 0);
    wait_group_events(1, &e2);
}
