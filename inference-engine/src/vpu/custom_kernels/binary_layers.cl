// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

__kernel void quantize(const __global half* __restrict src,
                       const __global half* __restrict input_low,
                       const __global half* __restrict input_high,
                       const __global half* __restrict output_low,
                       const __global half* __restrict output_high,
                       const __global half* __restrict dst,
                       int levels,
                       int input_low_size,
                       int input_high_size,
                       int output_low_size,
                       int output_high_size,
                       int W,
                       int H,
                       const __local half* __restrict src_local,
                             __local half* __restrict dst_local)
{

    int c = get_global_id(2);
    int C = get_global_size(2);

    half h_ilow  = (input_low_size   == 1 ? input_low[0]   : input_low[c]);
    half h_ihigh = (input_high_size  == 1 ? input_high[0]  : input_high[c]);
    half h_olow  = (output_low_size  == 1 ? output_low[0]  : output_low[c]);
    half h_ohigh = (output_high_size == 1 ? output_high[0] : output_high[c]);

    half const1 = (half)(0.01 > (h_ihigh - h_ilow) ? 0.0f : convert_float(levels - 1) / (convert_float(h_ihigh) - convert_float(h_ilow)));
    half const2 = (half)(!(levels - 1) ? 0.0f : (convert_float(h_ohigh) - convert_float(h_olow)) / convert_float(levels - 1));

    for (int h = 0; h < H; h++)
    {
        __local const half* __restrict addr_src = src_local + h*W;
        __local       half* __restrict addr_dst = dst_local + h*W;

        for (int w = 0; w < W / 8; w++)
        {
            half8 val = *((__local half8*)addr_src + w);
#if 1
            // round is too slow =( 902 b of code
            //half8 aux = round((val - (half8)h_ilow) * (half8)const1);

            half8 aux = (val - (half8)h_ilow) * (half8)const1 + (half8)0.5h;

            aux = (half8){
                (half)(short)(aux.s0),
                (half)(short)(aux.s1),
                (half)(short)(aux.s2),
                (half)(short)(aux.s3),
                (half)(short)(aux.s4),
                (half)(short)(aux.s5),
                (half)(short)(aux.s6),
                (half)(short)(aux.s7)
            };

            aux = aux * (half8)const2 + (half8)h_olow;

            // vector comparison add 756 b of assembly, so do in manually
            // short8 a = val <= (half8)h_olow;
            // short8 b = val >  (half8)h_ohigh;

            short8 a;
            short8 b;
            a.s0 = (val.s0 <= h_ilow);
            a.s1 = (val.s1 <= h_ilow);
            a.s2 = (val.s2 <= h_ilow);
            a.s3 = (val.s3 <= h_ilow);
            a.s4 = (val.s4 <= h_ilow);
            a.s5 = (val.s5 <= h_ilow);
            a.s6 = (val.s6 <= h_ilow);
            a.s7 = (val.s7 <= h_ilow);

            b.s0 = (val.s0 > h_ihigh);
            b.s1 = (val.s1 > h_ihigh);
            b.s2 = (val.s2 > h_ihigh);
            b.s3 = (val.s3 > h_ihigh);
            b.s4 = (val.s4 > h_ihigh);
            b.s5 = (val.s5 > h_ihigh);
            b.s6 = (val.s6 > h_ihigh);
            b.s7 = (val.s7 > h_ihigh);

            a = ~(a-(short8)1);
            b = ~(b-(short8)1);

            short8 c1 = (~a &  b);
            short8 c2 = (~a & ~b);

            short8 res = a  & as_short8((half8)h_olow)
                       | c1 & as_short8((half8)h_ohigh)
                       | c2 & as_short8(aux);

            *((__local half8*)addr_dst + w) = as_half8(res);
#else
            *((__local half8*)addr_dst + w) = val;
#endif
        }
        for (int w = W & (~0x7); w < W; w++)
        {
            half val = addr_src[w];
#if 1
            short a = val <= h_ilow; a = ~(a-1);
            short b = val > h_ihigh; b = ~(b-1);

            short c1 = (~a &  b);
            short c2 = (~a & ~b);

            short res = a  & as_short(h_olow)
                      | c1 & as_short(h_ohigh)
                      | c2 & as_short(((half)(round( (val - h_ilow) * const1) * const2) + h_olow));

            addr_dst[w] = as_half(res);
#else
            addr_dst[w] = val;
#endif
        }
    }
}
__kernel void __dma_preload_quantize(const __global half* __restrict src,
                       const __global half* __restrict input_low,
                       const __global half* __restrict input_high,
                       const __global half* __restrict output_low,
                       const __global half* __restrict output_high,
                       const __global half* __restrict dst,
                       int levels,
                       int input_low_size,
                       int input_high_size,
                       int output_low_size,
                       int output_high_size,
                       int W,
                       int H,
                             __local half* __restrict src_local,
                       const __local half* __restrict dst_local)
{
    const int sizePlane = W*H;
    async_work_group_copy(src_local ,src + get_group_id(2)*sizePlane, sizePlane, 0);
}
__kernel void __dma_postwrite_quantize(const __global half* __restrict src,
                       const __global half* __restrict input_low,
                       const __global half* __restrict input_high,
                       const __global half* __restrict output_low,
                       const __global half* __restrict output_high,
                             __global half* __restrict dst,
                       int levels,
                       int input_low_size,
                       int input_high_size,
                       int output_low_size,
                       int output_high_size,
                       int W,
                       int H,
                       const __local half* __restrict src_local,
                       const __local half* __restrict dst_local)
{
    const int sizePlane = W*H;
    async_work_group_copy(dst + get_group_id(2)*sizePlane ,dst_local,  sizePlane, 0);
}

__kernel void binarization(const __global half* __restrict src,
                           const __global half* __restrict input_low_high,
                           const __global half* __restrict dst,
                                          int              switch_out,
                                          int              input_low_high_size,
                                          int              W,
                                          int              H,
                           const __local  half* __restrict src_local,
                                 __local  half* __restrict dst_local)
{
    int c = get_global_id(2);
    int C = get_global_size(2);

    half dst_low  = switch_out ? 1.h : -1.h;
    half dst_high = switch_out ? -1.h : 1.h;

    half s_ilow_ihigh  = input_low_high_size == 1 ? input_low_high[0] : input_low_high[c];

    for (int h = 0; h < H; h++) {

        __local const half* __restrict addr_src = src_local + h*W;
        __local       half* __restrict addr_dst = dst_local + h*W;

#if 1
        for (int w = 0; w < W / 8; w++) {

            half8 h_src_val8 = (*((__local half8*)addr_src + w));

            short8 cond1;
            cond1.s0 = (h_src_val8.s0 <= s_ilow_ihigh);
            cond1.s1 = (h_src_val8.s1 <= s_ilow_ihigh);
            cond1.s2 = (h_src_val8.s2 <= s_ilow_ihigh);
            cond1.s3 = (h_src_val8.s3 <= s_ilow_ihigh);
            cond1.s4 = (h_src_val8.s4 <= s_ilow_ihigh);
            cond1.s5 = (h_src_val8.s5 <= s_ilow_ihigh);
            cond1.s6 = (h_src_val8.s6 <= s_ilow_ihigh);
            cond1.s7 = (h_src_val8.s7 <= s_ilow_ihigh);

            cond1 = ~(cond1-(short8)1);

            short8 res = cond1 & as_short8((half8)dst_low) | ~cond1 & as_short8((half8)dst_high);

            *((__local half8*)addr_dst + w) = as_half8(res);
        }
#endif
        for (int w = W & (~0x7); w < W; w++)
        {
            addr_dst[w] = (addr_src[w] <= s_ilow_ihigh) ? dst_low : dst_high;
        }
    }
}
__kernel void __dma_preload_binarization(const __global half* __restrict src,
                           const __global half* __restrict input_low_high,
                           const __global half* __restrict dst,
                                          int              switch_out,
                                          int              input_low_high_size,
                                          int              W,
                                          int              H,
                                 __local  half* __restrict src_local,
                           const __local  half* __restrict dst_local)
{
    const int sizePlane = W*H;
    async_work_group_copy(src_local ,src + get_group_id(2)*sizePlane, sizePlane, 0);
}
__kernel void __dma_postwrite_binarization(const __global half* __restrict src,
                           const __global half* __restrict input_low_high,
                                 __global half* __restrict dst,
                                          int              switch_out,
                                          int              input_low_high_size,
                                          int              W,
                                          int              H,
                           const __local  half* __restrict src_local,
                           const __local  half* __restrict dst_local)
{
    const int sizePlane = W*H;
    async_work_group_copy(dst + get_group_id(2)*sizePlane ,dst_local,  sizePlane, 0);
}