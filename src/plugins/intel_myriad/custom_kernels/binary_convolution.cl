// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

int extract_weights(uchar val, int bit) { return ((val >> bit) & 1); }

__kernel void binary_convolution(
    const __global half *restrict src_data,
    const __global uchar *restrict weights_data,
    __global half *restrict dst_data,
    float pad_value,

    int IW,
    int IH,
    int IC,

    int DW,
    int DH,

    int GC,

    int KW,
    int KH,

    int PW,
    int PH,

    int SW,
    int SH)
{
    int ipad_value = ((pad_value > 0.f) ? 1 : 0);
    int c          = get_global_id(2);
    int y          = get_global_id(1);
    int x          = get_global_id(0);

    int OC = get_global_size(2);
    int OH = get_global_size(1);
    int OW = get_global_size(0);

    int KD = 1;
    int SD = 0;
    int DD = 0;
    int PD = 0;
    int ID = 1;
    int OD = 1;

    int nbits = 8;

    int g  = c % GC;
    int oc = c / GC;
    int oh = y;
    int ow = x;

    for (int od = 0; od < OD; od++) {
        int oidx = g * OC / GC * OD * OH * OW + oc * OD * OH * OW + od * OH * OW + oh * OW + ow;

        int res = 0;

        for (int ic = 0; ic < IC / GC; ic++) {
            for (int kd = 0; kd < KD; kd++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int widx = g * OC / GC * IC / GC * KD * KH * KW
                                   + oc * IC / GC * KD * KH * KW + ic * KD * KH * KW + kd * KH * KW
                                   + kh * KW + kw;

                        int w = extract_weights(weights_data[widx / nbits], (widx % nbits));

                        int s;

                        int iw = ow * SW - PW + kw * DW;
                        int ih = oh * SH - PH + kh * DH;
                        int id = od * SD - PD + kd * DD;

                        if (iw < 0 || iw >= (int)IW || ih < 0 || ih >= (int)IH || id < 0
                            || id >= (int)ID) {
                            s = ipad_value;
                        } else {
                            int iidx = g * IC / GC * ID * IH * IW + ic * ID * IH * IW + id * IH * IW
                                       + ih * IW + iw;

                            s = ((src_data[iidx] > 0.f) ? 1 : 0);
                        }

                        res += s ^ w;
                    }
                }
            }
        }

        dst_data[oidx] = (half)(IC / GC * KD * KH * KW - 2 * res);
    }
}
