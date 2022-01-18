// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

__kernel void Convolution1x1_NHWC(
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

    const int sizeAct = IW * IC;

    event_t e1 = async_work_group_copy(in_local, in + get_group_id(0) * sizeAct, sizeAct, 0);
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
        half8 val8_1 = 0.0f;
        half8 val8_2 = 0.0f;
        half8 val8_3 = 0.0f;
        half8 val8_4 = 0.0f;
        half8 val8_5 = 0.0f;
        half8 val8_6 = 0.0f;
        half8 val8_7 = 0.0f;

        __local half8 *in8_0 = (__local half8 *)(&in_local[(iw + 0) * IC]);
        __local half8 *in8_1 = (__local half8 *)(&in_local[(iw + 1) * IC]);
        __local half8 *in8_2 = (__local half8 *)(&in_local[(iw + 2) * IC]);
        __local half8 *in8_3 = (__local half8 *)(&in_local[(iw + 3) * IC]);
        __local half8 *in8_4 = (__local half8 *)(&in_local[(iw + 4) * IC]);
        __local half8 *in8_5 = (__local half8 *)(&in_local[(iw + 5) * IC]);
        __local half8 *in8_6 = (__local half8 *)(&in_local[(iw + 6) * IC]);
        __local half8 *in8_7 = (__local half8 *)(&in_local[(iw + 7) * IC]);

        for (uint ic = 0; ic < IC / 8; ++ic) {
            val8_0 += (in8_0[ic]) * (w8[ic]);
            val8_1 += (in8_1[ic]) * (w8[ic]);
            val8_2 += (in8_2[ic]) * (w8[ic]);
            val8_3 += (in8_3[ic]) * (w8[ic]);
            val8_4 += (in8_4[ic]) * (w8[ic]);
            val8_5 += (in8_5[ic]) * (w8[ic]);
            val8_6 += (in8_6[ic]) * (w8[ic]);
            val8_7 += (in8_7[ic]) * (w8[ic]);
        }

        half val_0 = 0.0f;
        half val_1 = 0.0f;
        half val_2 = 0.0f;
        half val_3 = 0.0f;
        half val_4 = 0.0f;
        half val_5 = 0.0f;
        half val_6 = 0.0f;
        half val_7 = 0.0f;
        for (uint ic = IC & (~0x7); ic < IC; ++ic) {
            val_0 += *((__local half *)in8_0 + ic) * (*((__global half *)w8 + ic));
            val_1 += *((__local half *)in8_1 + ic) * (*((__global half *)w8 + ic));
            val_2 += *((__local half *)in8_2 + ic) * (*((__global half *)w8 + ic));
            val_3 += *((__local half *)in8_3 + ic) * (*((__global half *)w8 + ic));
            val_4 += *((__local half *)in8_4 + ic) * (*((__global half *)w8 + ic));
            val_5 += *((__local half *)in8_5 + ic) * (*((__global half *)w8 + ic));
            val_6 += *((__local half *)in8_6 + ic) * (*((__global half *)w8 + ic));
            val_7 += *((__local half *)in8_7 + ic) * (*((__global half *)w8 + ic));
        }
        out_local[ow + 0] = __builtin_shave_sau_sumx_f16_r(val8_0) + val_0;
        out_local[ow + 1] = __builtin_shave_sau_sumx_f16_r(val8_1) + val_1;
        out_local[ow + 2] = __builtin_shave_sau_sumx_f16_r(val8_2) + val_2;
        out_local[ow + 3] = __builtin_shave_sau_sumx_f16_r(val8_3) + val_3;
        out_local[ow + 4] = __builtin_shave_sau_sumx_f16_r(val8_4) + val_4;
        out_local[ow + 5] = __builtin_shave_sau_sumx_f16_r(val8_5) + val_5;
        out_local[ow + 6] = __builtin_shave_sau_sumx_f16_r(val8_6) + val_6;
        out_local[ow + 7] = __builtin_shave_sau_sumx_f16_r(val8_7) + val_7;
    }
    for (uint ow = (OW & (~0x7)); ow < OW; ow++) {

        uint iw = ow;
        uint ih = oh;

        half8 val8 = 0.0f;

        __local half8 *in8 = (__local half8 *)(&in_local[iw * IC]);

        for (uint ic = 0; ic < IC / 8; ++ic) {
            val8 += (in8[ic]) * (w8[ic]);
        }

        half val = 0.0f;
        for (uint ic = (IC & (~0x7)); ic < IC; ++ic) {
            val += (*((__local half *)in8 + ic)) * (*((__global half *)w8 + ic));
        }
        out_local[ow] = __builtin_shave_sau_sumx_f16_r(val8) + val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy(
        out + get_group_id(1) * OW * OH + get_group_id(0) * OW,
        out_local,
        OW,
        0);
    wait_group_events(1, &e2);
}
