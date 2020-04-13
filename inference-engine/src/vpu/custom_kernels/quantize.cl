#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void __dma_preload_quantize(const __global half* __restrict src,
                                     const __global half* __restrict input_low,
                                     const __global half* __restrict input_high,
                                     const __global half* __restrict output_low,
                                     const __global half* __restrict output_high,
                                     const __global half* __restrict dst,
                                                    int              levels,
                                                    int              input_low_size,
                                                    int              input_high_size,
                                                    int              output_low_size,
                                                    int              output_high_size,
                                                    int              W,
                                                    int              H,
                                           __local  half* __restrict src_local,
                                     const __local  half* __restrict dst_local)
{
    const int sizePlane = W*H;
    async_work_group_copy(src_local ,src + get_group_id(0)*sizePlane, sizePlane, 0);
}

__kernel void __dma_postwrite_quantize( const __global half* __restrict src,
                                        const __global half* __restrict input_low,
                                        const __global half* __restrict input_high,
                                        const __global half* __restrict output_low,
                                        const __global half* __restrict output_high,
                                              __global half* __restrict dst,
                                                       int              levels,
                                                       int              input_low_size,
                                                       int              input_high_size,
                                                       int              output_low_size,
                                                       int              output_high_size,
                                                       int              W,
                                                       int              H,
                                        const __local half* __restrict src_local,
                                        const __local half* __restrict dst_local)
{
    const int sizePlane = W*H;
    async_work_group_copy(dst + get_group_id(0)*sizePlane ,dst_local,  sizePlane, 0);
}

__kernel void quantize(const __global half* __restrict src,
                       const __global half* __restrict input_low,
                       const __global half* __restrict input_high,
                       const __global half* __restrict output_low,
                       const __global half* __restrict output_high,
                       const __global half* __restrict dst,
                                      int              levels,
                                      int              input_low_size,
                                      int              input_high_size,
                                      int              output_low_size,
                                      int              output_high_size,
                                      int              W,
                                      int              H,
                       const __local half* __restrict src_local,
                             __local half* __restrict dst_local)
{

    int c = get_global_id(0);

    int C = get_global_size(0);

    half h_ilow  = (input_low_size   == 1 ? input_low[0]   : input_low[c]);
    half h_ihigh = (input_high_size  == 1 ? input_high[0]  : input_high[c]);
    half h_olow  = (output_low_size  == 1 ? output_low[0]  : output_low[c]);
    half h_ohigh = (output_high_size == 1 ? output_high[0] : output_high[c]);

    half8 h_ilow8  = h_ilow;
    half8 h_ihigh8 = h_ihigh;
    half8 h_olow8  = h_olow;
    half8 h_ohigh8 = h_ohigh;

    float f_ilow  = convert_float(h_ilow);
    float f_ihigh = convert_float(h_ihigh);
    float f_olow  = convert_float(h_olow);
    float f_ohigh = convert_float(h_ohigh);

    float8 f_ilow8  = f_ilow;
    float8 f_ihigh8 = f_ihigh;
    float8 f_olow8  = f_olow;
    float8 f_ohigh8 = f_ohigh;

    float const1 = !(f_ihigh - f_ilow) ? 0.0f : convert_float(levels - 1) / (f_ihigh - f_ilow);
    float const2 = !(levels - 1)       ? 0.0f : (f_ohigh - f_olow) / convert_float(levels - 1);
    
    for (int h = 0; h < H; h++) {
        int idx = h*W;

        __local half* addr_src = (__local half*)src_local + idx;
        __local half* addr_dst = (__local half*)dst_local + idx;

        for (int w = 0; w < W / 8; w++) {
            half8 h_src_val8 = (*((__local half8*)addr_src + w));
            float8 f_src_val8 = convert_float8(h_src_val8);

            short8 aux_cond1;
            aux_cond1.s0 = (h_src_val8.s0 <= h_ilow);
            aux_cond1.s1 = (h_src_val8.s1 <= h_ilow);
            aux_cond1.s2 = (h_src_val8.s2 <= h_ilow);
            aux_cond1.s3 = (h_src_val8.s3 <= h_ilow);
            aux_cond1.s4 = (h_src_val8.s4 <= h_ilow);
            aux_cond1.s5 = (h_src_val8.s5 <= h_ilow);
            aux_cond1.s6 = (h_src_val8.s6 <= h_ilow);
            aux_cond1.s7 = (h_src_val8.s7 <= h_ilow);
            aux_cond1 *= aux_cond1;

            short8 aux_cond2;
            aux_cond2.s0 = (h_src_val8.s0 > h_ihigh);
            aux_cond2.s1 = (h_src_val8.s1 > h_ihigh);
            aux_cond2.s2 = (h_src_val8.s2 > h_ihigh);
            aux_cond2.s3 = (h_src_val8.s3 > h_ihigh);
            aux_cond2.s4 = (h_src_val8.s4 > h_ihigh);
            aux_cond2.s5 = (h_src_val8.s5 > h_ihigh);
            aux_cond2.s6 = (h_src_val8.s6 > h_ihigh);
            aux_cond2.s7 = (h_src_val8.s7 > h_ihigh);
            aux_cond2 *= aux_cond2;

            short8 aux_cond3 = (!aux_cond1 & aux_cond2);
            short8 aux_cond4 = (!aux_cond1 & !aux_cond2);
            aux_cond3 *= aux_cond3;
            aux_cond4 *= aux_cond4;

            half8 cond1 = convert_half8(aux_cond1);
            half8 cond2 = convert_half8(aux_cond2);
            half8 cond3 = convert_half8(aux_cond3);
            half8 cond4 = convert_half8(aux_cond4);

            half8 aux;
            aux = convert_half8(round(((f_src_val8 - f_ilow8) * (float8)const1)) * (float8)const2 + f_olow8);
            half8 dst_val = (
                      (h_olow8  * cond1) +
                      (h_ohigh8 * cond3) +
                      (aux      * cond4)
                    );
            *((__local half8*)addr_dst + w) = dst_val;
        }

        for (int w = W & (~0x7); w < W; w++) {
            half h_src_val = addr_src[w];
            float f_src_val = convert_float(h_src_val);
            half dst_val;

            if (h_src_val <= h_ilow) {
                dst_val = h_olow;
            } else if (h_src_val > h_ihigh) {
                dst_val = h_ohigh;
            } else {
                dst_val = convert_half(round((f_src_val - f_ilow) * const1) * const2 + f_olow);
            }
            addr_dst[w] = dst_val;
        }
    }
}
