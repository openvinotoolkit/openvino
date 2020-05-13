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
#ifdef USE_OPTIMIZED_ROUND
    return (int)((ox + 0.5f) / f);
#else
    return ROUND((ox + 0.5f) / f - 0.5f);
#endif
}

static inline float triangleCoeff(float x)
{
    return 1.0f - fabs(x);
}

static inline float4 triangleCoeff4(float4 x)
{
    return 1.0f - fabs(x);
}

static inline half triangleCoeffHalf(half x)
{
    return 1.0h - fabs(x);
}

static inline half4 triangleCoeffHalf4(half4 x)
{
    return 1.0h - fabs(x);
}

static inline half8 triangleCoeffHalf8(half8 x)
{
    return 1.0h - fabs(x);
}

#define USE_MANUAL_DMA

#if defined (USE_MANUAL_DMA)

__kernel void __dma_preload_resample_with_antialias(__global const half* restrict src,
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
    const int r = (factor > 1.0f) ? 2 : ceil(1.0f / factor);
    const int oy_first = get_group_id(1) * get_local_size(1);
    const int oy_last  = (get_group_id(1) + 1) * get_local_size(1) - 1;
    const int iy_first = max(out_to_in(oy_first, factor) - r, 0);
    const int iy_last  = min(out_to_in(oy_last, factor) + r, ih - 1);
    const int iy_size = iy_last - iy_first + 1;

    WorkGroupDmaCreateStrideTransaction(
        src + get_group_id(2)*get_local_size(2)*ih*iw + iy_first*iw, // src
        local_src, // dst
        iy_size * iw * sizeof(half), // src_width,
        iy_size * iw * sizeof(half), // dst_width,
        ih * iw * sizeof(half), // src_stride,
        iy_size * iw * sizeof(half), // dst_stride,
        get_local_size(2) * iy_size * iw * sizeof(half), // size
        0);
}

__kernel void __dma_postwrite_resample_with_antialias(__global const half* restrict _0,
                                               __global       half* restrict dst,
                                               __local        half* restrict _1,
                                               __local        half* restrict dst_local,
                                               int iw,
                                               int ih,
                                               float factor,
                                               int ow,
                                               int oh,
                                               int channels)
{
    WorkGroupDmaCreateStrideTransaction(
        dst_local,  // src
        dst + get_group_id(2)*get_local_size(2)*get_global_size(1)*ow + get_group_id(1)*get_local_size(1)*ow,  // dst
        get_local_size(1) * ow * sizeof(half), // src_width,
        get_local_size(1) * ow * sizeof(half), // dst_width,
        get_local_size(1) * ow * sizeof(half), // src_stride,
        get_global_size(1) * ow * sizeof(half), // dst_stride,
        get_local_size(2) * get_local_size(1) * ow * sizeof(half), // size
        0);
}

__kernel void resample_with_antialias(const __global half* restrict src,
                                      __global half* restrict dst,
                                      __local half* restrict local_src,
                                      __local half* restrict local_dst,
                                      int iw,
                                      int ih,
                                      float factor,
                                      int ow,
                                      int oh,
                                      int channels)
{
    const int r = (factor > 1.0f) ? 2 : ceil(1.0f / factor);
    const int oy_first = get_group_id(1) * get_local_size(1);
    const int oy_last  = (get_group_id(1) + 1) * get_local_size(1) - 1;
    const int iy_first = max(out_to_in(oy_first, factor) - r, 0);
    const int iy_last  = min(out_to_in(oy_last, factor) + r, ih - 1);
    const int iy_size = iy_last - iy_first + 1;
    const int oy = get_global_id(1);
    const float iy_f = ((oy + 0.5f) / factor - 0.5f) - iy_first;
    const int iy = ROUND(iy_f);

    __local half const *restrict start_src = local_src + iw * get_local_id(1) + iw * iy_size * get_local_id(2);
    __local half       *restrict start_dst = local_dst + ow * get_local_id(1) + ow * get_local_size(1) * get_local_id(2);

    for (int ox = 0; ox < ow; ox++)
    {
        const float ix_f = (float)((ox + 0.5f) / factor) - 0.5f;
        const int ix_i = ROUND(ix_f);

        float4 v_sum = 0.f;
        float4 v_wsum = 0.f;
        for (int y = 0; y < iy_size; y++)
        {
            float dy = iy_f - y;
            int x = max(ix_i - r, 0);
            int end_x = min(ix_i + r, iw - 1);

            float4 dx;
            for (int i = 0; i < 4; i++)
                dx[i] = ix_f - x - i;

            for (; x < end_x - 3; x += 4, dx -= 4)
            {
                float4 w = factor*triangleCoeff4(factor*dx) * factor*triangleCoeff(factor*dy);
                float4 src_vec = { start_src[y*iw + x + 0],
                                   start_src[y*iw + x + 1],
                                   start_src[y*iw + x + 2],
                                   start_src[y*iw + x + 3] };

                v_sum += w * src_vec;
                v_wsum += w;
            }

            for (; x <= end_x; x++)
            {
                float dx = ix_f - x;
                float w = factor*triangleCoeff(factor*dx) * factor*triangleCoeff(factor*dy);

                v_sum[0] += w * start_src[y*iw + x];
                v_wsum[0] += w;
            }
        }

        v_sum[0] = v_sum[0] + v_sum[1] + v_sum[2] + v_sum[3];
        v_wsum[0] = v_wsum[0] + v_wsum[1] + v_wsum[2] + v_wsum[3];

        start_dst[get_local_id(1)*ow + ox] = (!v_wsum[0]) ? 0.0f : (half)(v_sum[0] / v_wsum[0]);
    }
}

#else

__kernel void resample_with_antialias(const __global half* restrict src,
                                      __global half* restrict dst,
                                      __local half* restrict _0,
                                      __local half* restrict _1,
                                      int iw,
                                      int ih,
                                      float factor,
                                      int ow,
                                      int oh,
                                      int channels)
{
    int oy = get_global_id(1);
    int c = get_global_id(2);

    int r = (factor > 1.0f) ? 2 : ceil((1.0f)/factor);

    const __global half* restrict start_src = src + iw * ih * c;
    __global half* restrict start_dst = dst + ow * oh * c;

    float iy_f = (oy + 0.5) / factor - 0.5f;
    int iy_i = ROUND(iy_f);

    for (int ox = 0; ox < ow; ox++)
    {
        float ix_f = (ox + 0.5) / factor - 0.5f;
        int ix_i = ROUND(ix_f);

        float4 v_sum = 0.f;
        float4 v_wsum = 0.f;

        for (int y = max(iy_i - r, 0); y <= min(iy_i + r, (int)ih - 1); y++)
        {
            float dy = iy_f - y;
            int x = max(ix_i - r, 0);
            int end_x = min(ix_i + r, (int)iw - 1);

            float4 dx;
            for (int i = 0; i < 4; i++)
                dx[i] = ix_f - x - i;

            for (; x <= end_x - 3; x += 4, dx -= 4)
            {
                float4 w = factor*triangleCoeff4(factor*dx) * factor*triangleCoeff(factor*dy);
                float4 src_vec = { start_src[y*iw + x + 0],
                                   start_src[y*iw + x + 1],
                                   start_src[y*iw + x + 2],
                                   start_src[y*iw + x + 3] };

                v_sum += w * src_vec;
                v_wsum += w;
            }

            for (; x <= end_x; x++)
            {
                float dx = ix_f - x;
                float w = factor*triangleCoeff(factor*dx) * factor*triangleCoeff(factor*dy);

                v_sum[0] += w * start_src[y*iw + x];
                v_wsum[0] += w;
            }
        }

        v_sum[0] = v_sum[0] + v_sum[1] + v_sum[2] + v_sum[3];
        v_wsum[0] = v_wsum[0] + v_wsum[1] + v_wsum[2] + v_wsum[3];

        start_dst[oy*ow + ox] = (!v_wsum[0]) ? (half)0.0f : (half)(v_sum[0] / v_wsum[0]);
    }
}

#endif
