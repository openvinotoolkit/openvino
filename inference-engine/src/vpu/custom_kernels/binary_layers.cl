// Copyright (C) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

int extract_weights(uchar val, int bit) {
  return ((val >> bit) & 1);
}

__kernel void binary_convolution(const __global half* restrict src_data,
                                 const __global uchar* restrict weights_data,
                                 __global half* restrict dst_data,
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
    int c = get_global_id(2);
    int y = get_global_id(1);
    int x = get_global_id(0);

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
        int oidx = g  * OC / GC * OD * OH * OW
                 + oc * OD * OH * OW
                 + od * OH * OW
                 + oh * OW
                 + ow;

        int res = 0;

        for (int ic = 0; ic < IC / GC; ic++) {
            for (int kd = 0; kd < KD; kd++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int widx = g  * OC / GC * IC / GC * KD * KH * KW
                                 + oc * IC / GC * KD * KH * KW
                                 + ic * KD * KH * KW
                                 + kd * KH * KW
                                 + kh * KW
                                 + kw;

                        int w = extract_weights(weights_data[widx/nbits], (widx % nbits));

                        int s;

                        int iw = ow * SW - PW + kw * DW;
                        int ih = oh * SH - PH + kh * DH;
                        int id = od * SD - PD + kd * DD;

                        if (iw < 0 || iw >= (int) IW ||
                            ih < 0 || ih >= (int) IH ||
                            id < 0 || id >= (int) ID) {
                            s = ipad_value;
                        } else {
                            int iidx = g  * IC / GC * ID * IH * IW
                                     + ic * ID * IH * IW
                                     + id * IH * IW
                                     + ih * IW
                                     + iw;

                            s = ((src_data[iidx] > 0.f) ? 1 : 0);
                        }

                        res += s ^ w;
                    }
                }
            }
        }

        dst_data[oidx] = (half)(IC/GC*KD*KH*KW - 2*res);
    }
}

__kernel void quantize(const __global half* restrict src,
                       const __global half* restrict input_low,
                       const __global half* restrict input_high,
                       const __global half* restrict output_low,
                       const __global half* restrict output_high,
                       __global half* restrict dst,
                       int levels,
                       int input_low_size,
                       int input_high_size,
                       int output_low_size,
                       int output_high_size)
{
    int c = get_global_id(2);
    int h = get_global_id(1);
    int w = get_global_id(0);

    int C = get_global_size(2);
    int H = get_global_size(1);
    int W = get_global_size(0);

    int idx = c*H*W + h*W + w;

    half ilow  = (input_low_size   == 1 ? input_low[0]   : input_low[c]);
    half ihigh = (input_high_size  == 1 ? input_high[0]  : input_high[c]);
    half olow  = (output_low_size  == 1 ? output_low[0]  : output_low[c]);
    half ohigh = (output_high_size == 1 ? output_high[0] : output_high[c]);

    if (src[idx] <= ilow)
    {
        dst[idx] = olow;
    }
    else if (src[idx] > ihigh)
    {
        dst[idx] = ohigh;
    }
    else
    {
        dst[idx] = round((src[idx] - ilow) / (ihigh - ilow) * (levels-1)) / (levels-1) * (ohigh - olow) + olow;
    }
}
