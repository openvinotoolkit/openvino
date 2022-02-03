// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void Convolution1x1_NCHW(
    const __global half *in,
    const __global half *out,
    const __global half *w,
    int IW,
    int IH,
    int IC,
    int OW,
    int OH,
    int OC)
{
    __local half in_local[8 * 1024];
    __local half out_local[8 * 1024];

    event_t e1 = async_work_group_copy_2D2D(
        in_local, // dst
        in + get_group_id(0) * IW, // src
        IW, // num_elements_per_line,
        IC, // num_lines,
        IW * IH - IW, // src_line_stride,
        0, // dst_line_stride,
        0);
    wait_group_events(1, &e1);

    int oh = get_global_id(0);
    int oc = get_global_id(1);

    int stride;
    int write_output = 0;
    __global half *src;

    __global half8 *w8 = (__global half8 *)(&w[oc * IC]);
    __global half *w1  = (__global half *)(&w[oc * IC]);

    for (uint ow = 0; ow < (OW & (~0x7)); ow += 8) {
        uint iw = ow;
        uint ih = oh;

        half8 val8_0 = 0.0f;

        __local half8 *in8_0 = (__local half8 *)(&in_local[iw + 0 * IW]);
        __local half8 *in8_1 = (__local half8 *)(&in_local[iw + 1 * IW]);
        __local half8 *in8_2 = (__local half8 *)(&in_local[iw + 2 * IW]);
        __local half8 *in8_3 = (__local half8 *)(&in_local[iw + 3 * IW]);
        __local half8 *in8_4 = (__local half8 *)(&in_local[iw + 4 * IW]);
        __local half8 *in8_5 = (__local half8 *)(&in_local[iw + 5 * IW]);
        __local half8 *in8_6 = (__local half8 *)(&in_local[iw + 6 * IW]);
        __local half8 *in8_7 = (__local half8 *)(&in_local[iw + 7 * IW]);

        for (uint ic = 0; ic < IC / 8; ic++) {
            val8_0 += (in8_0[ic * IW]) * ((half8)w8[ic].s0);
            val8_0 += (in8_1[ic * IW]) * ((half8)w8[ic].s1);
            val8_0 += (in8_2[ic * IW]) * ((half8)w8[ic].s2);
            val8_0 += (in8_3[ic * IW]) * ((half8)w8[ic].s3);
            val8_0 += (in8_4[ic * IW]) * ((half8)w8[ic].s4);
            val8_0 += (in8_5[ic * IW]) * ((half8)w8[ic].s5);
            val8_0 += (in8_6[ic * IW]) * ((half8)w8[ic].s6);
            val8_0 += (in8_7[ic * IW]) * ((half8)w8[ic].s7);
        }

        for (uint ic = (IC & (~0x7)); ic < IC; ++ic) {
            val8_0 += *((__local half8 *)(&in_local[iw + ic * IW])) * ((half8)w1[ic]);
        }
        *((__local half8 *)&out_local[ow + 0]) = (val8_0);
    }

    uint iw = (OW & (~0x7));
    uint ih = oh;

    half8 val8_0 = 0.0f;

    __local half8 *in8_0 = (__local half8 *)(&in_local[iw + 0 * IW]);
    __local half8 *in8_1 = (__local half8 *)(&in_local[iw + 1 * IW]);
    __local half8 *in8_2 = (__local half8 *)(&in_local[iw + 2 * IW]);
    __local half8 *in8_3 = (__local half8 *)(&in_local[iw + 3 * IW]);
    __local half8 *in8_4 = (__local half8 *)(&in_local[iw + 4 * IW]);
    __local half8 *in8_5 = (__local half8 *)(&in_local[iw + 5 * IW]);
    __local half8 *in8_6 = (__local half8 *)(&in_local[iw + 6 * IW]);
    __local half8 *in8_7 = (__local half8 *)(&in_local[iw + 7 * IW]);

    for (uint ic = 0; ic < IC / 8; ic++) {
        val8_0 += (in8_0[ic * IW]) * ((half8)w8[ic].s0);
        val8_0 += (in8_1[ic * IW]) * ((half8)w8[ic].s1);
        val8_0 += (in8_2[ic * IW]) * ((half8)w8[ic].s2);
        val8_0 += (in8_3[ic * IW]) * ((half8)w8[ic].s3);
        val8_0 += (in8_4[ic * IW]) * ((half8)w8[ic].s4);
        val8_0 += (in8_5[ic * IW]) * ((half8)w8[ic].s5);
        val8_0 += (in8_6[ic * IW]) * ((half8)w8[ic].s6);
        val8_0 += (in8_7[ic * IW]) * ((half8)w8[ic].s7);
    }

    for (uint ic = (IC & (~0x7)); ic < IC; ++ic) {
        val8_0 += *((__local half8 *)(&in_local[iw + ic * IW])) * ((half8)w1[ic]);
    }
    for (uint ow = (OW & (~0x7)); ow < OW; ow++) {
        out_local[ow + 0] = (val8_0[ow % 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy(
        out + get_group_id(1) * OW * OH + get_group_id(0) * OW,
        out_local,
        OW,
        0);
    wait_group_events(1, &e2);
}
