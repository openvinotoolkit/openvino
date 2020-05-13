// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define USE_OPTIMIZED_ROUND

#ifdef USE_OPTIMIZED_ROUND
    #define ROUND(x)  ((int)((x) + 0.5f))
#else
    #define ROUND(x)  (int)(round(x))
#endif

inline int out_to_in(float ox, float f) {
    return (int)((ox + 0.5f) * f);
}

#define USE_MANUAL_DMA

#if defined (USE_MANUAL_DMA)

void interpolationCHW_nn(__local half* psrc, __local half* pdst, int OW, int IW, int C, float rw, float rh)
{
    float alpha = rh / 2.0f - 0.5f;

    for (int w = 0; w < OW/8; w++)
    {
        float fw0 = rw*(w*8+0) + alpha;
        float fw1 = rw*(w*8+1) + alpha;
        float fw2 = rw*(w*8+2) + alpha;
        float fw3 = rw*(w*8+3) + alpha;

        float fw4 = rw*(w*8+4) + alpha;
        float fw5 = rw*(w*8+5) + alpha;
        float fw6 = rw*(w*8+6) + alpha;
        float fw7 = rw*(w*8+7) + alpha;

        int iw0 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw0), IW-1);
        int iw1 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw1), IW-1);
        int iw2 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw2), IW-1);
        int iw3 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw3), IW-1);

        int iw4 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw4), IW-1);
        int iw5 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw5), IW-1);
        int iw6 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw6), IW-1);
        int iw7 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw7), IW-1);

        for (int c = 0; c < C; c++)
        {
            half8 val = {
                *((__local half*)(psrc + c * IW + iw0)),
                *((__local half*)(psrc + c * IW + iw1)),

                *((__local half*)(psrc + c * IW + iw2)),
                *((__local half*)(psrc + c * IW + iw3)),

                *((__local half*)(psrc + c * IW + iw4)),
                *((__local half*)(psrc + c * IW + iw5)),

                *((__local half*)(psrc + c * IW + iw6)),
                *((__local half*)(psrc + c * IW + iw7)),
            };
            *((__local half8*)(pdst + c * OW + w*8)) = val;
        }
    }

    for (int w = OW/8*8; w < OW; w++)
    {
        float fw = rw*w + alpha;
        int iw0 = __builtin_shave_cmu_min_i32_rr_int((int)ROUND(fw), IW-1);

        for (int c = 0; c < C; c++)
        {
            *((__local half*)(pdst + c * OW + w)) = *((__local half*)(psrc + c * IW + iw0));
        }
    }
}

__kernel void __dma_preload_resample_nearest(__global const half* restrict src,
                                             __global       half* restrict _0,
                                             __local        half* restrict local_src,
                                             __local        half* restrict _1,
                                             int iw,
                                             int ih,
                                             float factor,
                                             int ow,
                                             int oh,
                                             int channels)
{
    const int oy_first = get_group_id(1) * get_local_size(1);
    const int oy_last = (get_group_id(1) + 1) * get_local_size(1) - 1;
    const int iy_first = out_to_in(oy_first, 1.0 / factor);
    const int iy_last = out_to_in(oy_last, 1.0 /factor);
    const int iy_size = iy_last - iy_first + 1;

    WorkGroupDmaCreateStrideTransaction(
        src + get_group_id(2)*channels*ih*iw + iy_first*iw, // src
        local_src, // dst
        iy_size * iw * sizeof(half), // src_width,
        iy_size * iw * sizeof(half), // dst_width,
        ih * iw * sizeof(half), // src_stride,
        iy_size * iw * sizeof(half), // dst_stride,
        channels * iy_size * iw * sizeof(half), // size
        0);
}

__kernel void __dma_postwrite_resample_nearest(__global const half* restrict _0,
                                               __global       half* restrict dst,
                                               __local        half* restrict _1,
                                               __local        half* restrict local_dst,
                                               int iw,
                                               int ih,
                                               float factor,
                                               int ow,
                                               int oh,
                                               int channels)
{

    WorkGroupDmaCreateStrideTransaction(
        local_dst,  // src
        dst + get_group_id(2)*channels*get_global_size(1)*ow + get_group_id(1)*get_local_size(1)*ow,  // dst
        get_local_size(1) * ow * sizeof(half), // src_width,
        get_local_size(1) * ow * sizeof(half), // dst_width,
        get_local_size(1) * ow * sizeof(half), // src_stride,
        get_global_size(1) * ow * sizeof(half), // dst_stride,
        channels * get_local_size(1) * ow * sizeof(half), // size
        0);
}

kernel void resample_nearest(__global const half* restrict src,
                             __global       half* restrict dst,
                             __local        half* restrict local_src,
                             __local        half* restrict local_dst,
                             int iw,
                             int ih,
                             float factor,
                             int ow,
                             int oh,
                             int channels)
{
    interpolationCHW_nn(local_src, local_dst, ow, iw, channels, 1.0 / factor, 1.0 / factor);
}

#else // defined (USE_MANUAL_DMA)

kernel void resample_nearest(__global const half* restrict src,
                             __global       half* restrict dst,
                             __local        half* restrict local_src,
                             __local        half* restrict local_dst,
                             int iw,
                             int ih,
                             float factor,
                             int ow,
                             int oh,
                             int channels)
{
    const float inv_factor = 1.0f / factor;
    const int iy = out_to_in(get_global_id(1), inv_factor);

    __global half* dst_data = dst + get_global_id(1)*ow;
    __global half* src_data = src + iy*iw;

    for (int ox = 0; ox < ow; ++ox)
    {
        const int ix = out_to_in(ox, inv_factor);
        for (int c = 0; c < channels; c++) {
            dst_data[c*oh*ow + ox] = src_data[c*ih*iw + ix];
        }
    }
}

#endif // defined (USE_MANUAL_DMA)
