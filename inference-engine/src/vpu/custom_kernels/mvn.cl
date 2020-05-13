// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Define if runtime supports it. MX runtime is compatible, KMB is in WIP state
#define USE_MANUAL_DMA 1

// Set to 1 if only output is zerroed before kernel execution
#define USE_ATOMICS 0

void atomic_add_global(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

#if defined (USE_MANUAL_DMA)

__kernel void __dma_preload_reduction_mean(const __global half*  restrict src,
                                                 __global float* restrict mean,
                                                 __global float* restrict variance,
                                           int W,
                                           int H,
                                           int across_channels,
                                           __local half* restrict src_line)
{
    WorkGroupDmaCreateStrideTransaction(
        src + get_group_id(1)*get_local_size(1)*W +
              get_group_id(2)*get_local_size(2)*W*get_global_size(1), // src
        src_line,                                                     // dst
        W*get_local_size(1) * sizeof(half), // src width
        W*get_local_size(1) * sizeof(half), // dst width
        W*get_global_size(1) * sizeof(half), // src stride
        W*get_local_size(1) * sizeof(half),  // dst stride
        W*get_local_size(1)*get_local_size(2)*sizeof(half), //total size
        0
        );
}

__kernel void reduction_mean(const __global half*  restrict src,
                                   __global float* restrict mean,
                                   __global float* restrict variance,
                            int W,
                            int H,
                            int across_channels,
                            __local half* restrict src_line)
{
    int h = get_global_id(1);
    int c = get_global_id(2);

    const int MAX_LOCAL_SIZE = 8;

    __local float mbuf[MAX_LOCAL_SIZE];
    __local float vbuf[MAX_LOCAL_SIZE];

    mbuf[get_local_id(1)] = 0;
    vbuf[get_local_id(1)] = 0;

    if (h < H)
    {
        float sum  = 0.f;
        float sum2 = 0.f;

        float8 sum4  = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float8 sum24 = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        const __local half8* lsrc = ((const __local half8*)(src_line + get_local_id(1)*W) );

        #pragma unroll 16
        for (size_t w = 0; w < W/8; w++)
        {
            half8 sh = lsrc[w];
            float8 valf = convert_float8(sh);

            sum4 += valf;
            sum24 += valf*valf;
        }

        for (size_t w = W/8*8; w < W; w++)
        {
            float val = (float)src_line[get_local_id(1)*W + w];
            sum  += val;
            sum2 += val*val;
        }

        mbuf[get_local_id(1)] = sum4.s0  + sum4.s1  + sum4.s2  + sum4.s3  + sum4.s4  + sum4.s5  + sum4.s6  + sum4.s7  + sum;
        vbuf[get_local_id(1)] = sum24.s0 + sum24.s1 + sum24.s2 + sum24.s3 + sum24.s4 + sum24.s5 + sum24.s6 + sum24.s7 + sum2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(1) == 0)
    {
        float res  = 0;
        float res2 = 0;

        for (int i = 0; i < get_local_size(1); i++)
        {
            res  += mbuf[i];
            res2 += vbuf[i];
        }

// requires memory reset before layer execution
#if USE_ATOMICS
        int idx = (across_channels == 0) ? c : 0;

        atomic_add_global(mean + idx, res);
        atomic_add_global(variance + idx, res2);
#else
        int idx = c*get_num_groups(1) + get_group_id(1);

        mean[idx] = res;
        variance[idx] = res2;
#endif
    }
}

__kernel void __dma_preload_mvn_scale(const __global half * restrict src,
                          __global half * restrict dst,
                          __global float * restrict mean_part,
                          __global float * restrict power_mean,
                          int W,
                          int H1,
                          int across_channels,
                          int normalize_variance,
                          int nparts,
                          __local half * restrict src_line,
                          __local half * restrict dst_line
                          )
{
    WorkGroupDmaCreateStrideTransaction(
        src + get_group_id(1)*get_local_size(1)*W +
              get_group_id(2)*get_local_size(2)*W*get_global_size(1), // src
        src_line,                                                     // dst
        W*get_local_size(1) * sizeof(half), // src width
        W*get_local_size(1) * sizeof(half), // dst width
        W*get_global_size(1) * sizeof(half), // src stride
        W*get_local_size(1) * sizeof(half),  // dst stride
        W*get_local_size(1)*get_local_size(2)*sizeof(half), //total size
        0
        );
}

__kernel void __dma_postwrite_mvn_scale(const __global half * restrict src,
                          __global half * restrict dst,
                          __global float * restrict mean_part,
                          __global float * restrict power_mean,
                          int W,
                          int H1,
                          int across_channels,
                          int normalize_variance,
                          int nparts,
                          __local half * restrict src_line,
                          __local half * restrict dst_line)
{
    WorkGroupDmaCreateStrideTransaction(
        dst_line,                                                     // src
        dst + get_group_id(1)*get_local_size(1)*W +
              get_group_id(2)*get_local_size(2)*W*get_global_size(1), // dst
        W*get_local_size(1) * sizeof(half), // src width
        W*get_local_size(1) * sizeof(half), // dst width
        W*get_local_size(1) * sizeof(half),  // dst stride
        W*get_global_size(1) * sizeof(half), // src stride
        W*get_local_size(1)*get_local_size(2)*sizeof(half), //total size
        0
        );
}

__kernel void mvn_scale(const __global half * restrict src,
                          __global half * restrict dst,
                          __global float * restrict mean_part,
                          __global float * restrict power_mean,
                          int W,
                          int H1,
                          int across_channels,
                          int normalize_variance,
                          int nparts,
                          __local half * restrict src_line,
                          __local half * restrict dst_line)
{
    int h = get_global_id(1);
    int H = get_global_size(1);

    // can we avoid this check and use min/max? We can pass number of groups just as a param.
//#if !USE_ATOMICS
//    if (h >= H1) return;
//#endif

    int c = get_global_id(2);
    int C = get_global_size(2);

    int idx     = (across_channels == 0) ? nparts*c : 0;
    float scale = (across_channels == 0) ?      H*W : H*W*C;

#if USE_ATOMICS
    float mean = mean_part[idx];
    float variance = power_mean[idx];
#else

    int total   = (across_channels == 0) ?   nparts : nparts*C;
    float mean = 0.f;
    float variance = 0.f;

    for (int i = 0; i < total; i++)
    {
        mean     += mean_part[idx+i];
        variance += power_mean[idx+i];
    }
#endif

    mean = mean/scale;
    variance = variance/scale;
    variance = variance - mean*mean;
    variance = native_sqrt(variance) + 1e-9f;

    half hmean = mean;
    half hvariance = (normalize_variance == 0) ? 1.f : (1.f / variance);

    const __local half8 * restrict src_data8 = (const __local half8 * restrict)(src_line + get_local_id(1)*W);
    __local half8 * restrict dst_data8 = (__local half8 * restrict)(dst_line + get_local_id(1)*W);

    #pragma unroll 16
    for (size_t w = 0; w < W/8; w++)
    {
        dst_data8[w] = (src_data8[w] - hmean) * hvariance;
    }
    for (size_t w = W/8*8; w < W; w++)
    {
        dst_line[get_local_id(1)*W + w] = (src_line[get_local_id(1)*W + w] - hmean) * hvariance;
    }
}

#else

__kernel void reduction_mean(const __global half*  restrict src,
                                   __global float* restrict mean,
                                   __global float* restrict variance,
                            int W,
                            int H,
                            int across_channels,
                            __local half* restrict src_line) // for compatimility with DMA kernel
{
    int h = get_global_id(1);
    int c = get_global_id(2);

    const int MAX_LOCAL_SIZE = 8;

    __local float mbuf[MAX_LOCAL_SIZE];
    __local float vbuf[MAX_LOCAL_SIZE];

    mbuf[get_local_id(1)] = 0;
    vbuf[get_local_id(1)] = 0;

    if (h < H)
    {
        float sum  = 0.f;
        float sum2 = 0.f;

        float8 sum4  = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float8 sum24 = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        const __global half8* src_line = (const __global half8 *)(src + c*H*W + h*W);

        #pragma unroll 16
        for (size_t w = 0; w < W/8; w++)
        {
            half8 sh = src_line[w];
            float8 valf = convert_float8(sh);

            sum4 += valf;
            sum24 += valf*valf;
        }

        for (size_t w = W/8*8; w < W; w++)
        {
            float val = (float)src[c*H*W + h*W + w];

            sum  += val;
            sum2 += val*val;
        }

        mbuf[get_local_id(1)] = sum4.s0  + sum4.s1  + sum4.s2  + sum4.s3  + sum4.s4  + sum4.s5  + sum4.s6  + sum4.s7  + sum;
        vbuf[get_local_id(1)] = sum24.s0 + sum24.s1 + sum24.s2 + sum24.s3 + sum24.s4 + sum24.s5 + sum24.s6 + sum24.s7 + sum2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(1) == 0)
    {
        float res  = 0;
        float res2 = 0;

        for (int i = 0; i < get_local_size(1); i++)
        {
            res  += mbuf[i];
            res2 += vbuf[i];
        }

// requires memory reset before layer execution
#if USE_ATOMICS
        int idx = (across_channels == 0) ? c : 0;

        atomic_add_global(mean + idx, res);
        atomic_add_global(variance + idx, res2);
#else
        int idx = c*get_num_groups(1) + get_group_id(1);

        mean[idx] = res;
        variance[idx] = res2;
#endif
    }
}

__kernel void mvn_scale(const __global half * restrict src_data,
                          __global half * restrict dst_data,
                          __global float * restrict mean_part,
                          __global float * restrict power_mean,
                          int W,
                          int H1,
                          int across_channels,
                          int normalize_variance,
                          int nparts,
                          __local half * restrict src_line,
                          __local half * restrict dst_line)
{
    int h = get_global_id(1);
    int H = get_global_size(1);

    // can we avoid this check and use min/max? We can pass number of groups just as a param.
//#if !USE_ATOMICS
//    if (h >= H1) return;
//#endif

    int c = get_global_id(2);
    int C = get_global_size(2);

    int idx     = (across_channels == 0) ? nparts*c : 0;
    float scale = (across_channels == 0) ?      H*W : H*W*C;

#if USE_ATOMICS
    float mean = mean_part[idx];
    float variance = power_mean[idx];
#else

    int total   = (across_channels == 0) ?   nparts : nparts*C;
    float mean = 0.f;
    float variance = 0.f;

    for (int i = 0; i < total; i++)
    {
        mean     += mean_part[idx+i];
        variance += power_mean[idx+i];
    }
#endif

    mean = mean/scale;
    variance = variance/scale;
    variance = variance - mean*mean;
    variance = native_sqrt(variance) + 1e-9f;

    half hmean = mean;
    half hvariance = (normalize_variance == 0) ? 1.f : (1.f / variance);

    const __global half8 * restrict src_data8 = (const __global half8 * restrict)(src_data + c*H*W + h*W);
    __global half8 * restrict dst_data8 = (__global half8 * restrict)(dst_data + c*H*W + h*W);

    #pragma unroll 16
    for (size_t w = 0; w < W/8; w++)
    {
        dst_data8[w] = (src_data8[w] - hmean) * hvariance;
    }
    for (size_t w = W/8*8; w < W; w++)
    {
        dst_data[c*H*W + h*W + w] = (src_data[c*H*W + h*W + w] - hmean) * hvariance;
    }
}

#endif // USE_MANUAL_DMA
