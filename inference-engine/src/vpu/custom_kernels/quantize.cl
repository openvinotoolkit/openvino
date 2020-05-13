// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void __dma_preload_quantize(__global half const *const restrict src,
                                     __global half const *const restrict input_low,
                                     __global half const *const restrict input_high,
                                     __global half const *const restrict output_low,
                                     __global half const *const restrict output_high,
                                     __global half       *const restrict dst,
                                     int levels,
                                     int input_low_size,
                                     int input_high_size,
                                     int output_low_size,
                                     int output_high_size,
                                     int W,
                                     int C,
                                     __local half       *const restrict local_src,
                                     __local half const *const restrict local_dst)
{
    WorkGroupDmaCreateStrideTransaction(
        src + get_group_id(1) * get_local_size(1) * W, // src
        local_src, // dst
        W * sizeof(half), // src_width,
        W * sizeof(half), // dst_width,
        get_global_size(1) * W * sizeof(half), // src_stride,
        W * sizeof(half), // dst_stride,
        W * C * sizeof(half), // size
        0);
}

__kernel void __dma_postwrite_quantize(__global half const *const restrict src,
                                       __global half const *const restrict input_low,
                                       __global half const *const restrict input_high,
                                       __global half const *const restrict output_low,
                                       __global half const *const restrict output_high,
                                       __global half       *const restrict dst,
                                       int levels,
                                       int input_low_size,
                                       int input_high_size,
                                       int output_low_size,
                                       int output_high_size,
                                       int W,
                                       int C,
                                       __local half const *const restrict local_src,
                                       __local half const *const restrict local_dst)
{
    WorkGroupDmaCreateStrideTransaction(
        local_dst, // src
        dst + get_group_id(1) * get_local_size(1) * W, // dst
        W * sizeof(half), // src_width,
        W * sizeof(half), // dst_width,
        W * sizeof(half), // src_stride,
        get_global_size(1) * W * sizeof(half), // dst_stride,
        W * C * sizeof(half), // size
        0);
}

__kernel void quantize(__global half const *const restrict src,
                       __global half const *const restrict input_low,
                       __global half const *const restrict input_high,
                       __global half const *const restrict output_low,
                       __global half const *const restrict output_high,
                       __global half const *const restrict dst,
                       int levels,
                       int input_low_size,
                       int input_high_size,
                       int output_low_size,
                       int output_high_size,
                       int W,
                       int C,
                       __local half const *const restrict local_src,
                       __local half       *const restrict local_dst)
{
    int h = get_global_id(1);
    int H = get_global_size(1);

    for (int c = 0; c < C; c++)
    {
        half h_ilow  = (input_low_size   == 1 ? input_low[0]   : input_low[c]);
        half h_ihigh = (input_high_size  == 1 ? input_high[0]  : input_high[c]);
        half h_olow  = (output_low_size  == 1 ? output_low[0]  : output_low[c]);
        half h_ohigh = (output_high_size == 1 ? output_high[0] : output_high[c]);

        half const1 = (half)(!(h_ihigh - h_ilow) ? 0.0f : convert_float(levels - 1) / (convert_float(h_ihigh) - convert_float(h_ilow)));
        half const2 = (half)(!(levels - 1)       ? 0.0f : (convert_float(h_ohigh) - convert_float(h_olow)) / convert_float(levels - 1));

        __local const half* restrict addr_src = local_src + c*W;
        __local       half* restrict addr_dst = local_dst + c*W;

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
        //for (int w = 0         ; w < W; w++)
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
