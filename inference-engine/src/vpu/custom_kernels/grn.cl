// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define USE_MANUAL_DMA 1

#if defined (USE_MANUAL_DMA)

__kernel void __dma_preload_grn_NCHW(
    __global const half* restrict src,
    __global       half* restrict dst,
    __local        half* restrict local_src,
    __local        half* restrict local_dst,
    int C,
    float bias)
{
    WorkGroupDmaCreate3DTransaction(
        src + get_group_id(0)*get_local_size(0)
            + get_group_id(1)*get_local_size(1)*get_global_size(0), // src
        local_src, // dst
        get_local_size(0) * sizeof(half), // src width
        get_local_size(0) * sizeof(half), // dst width
        get_global_size(0) * sizeof(half), // src stride
        get_local_size(0) * sizeof(half), // dst stride
        C, // num planes
        get_global_size(0) * get_global_size(1) * sizeof(half), // src plane stride
        get_local_size(0) * get_local_size(1) * sizeof(half), // dst plane stride
        get_local_size(0) * get_local_size(1) * sizeof(half), // plane size
        0);
}

__kernel void __dma_postwrite_grn_NCHW(
    __global const half* restrict src,
    __global       half* restrict dst,
    __local  const half* restrict local_src,
    __local        half* restrict local_dst,
    int C,
    float bias)
{
    WorkGroupDmaCreate3DTransaction(
        local_dst, // src
        dst + get_group_id(0)*get_local_size(0)
            + get_group_id(1)*get_local_size(1)*get_global_size(0), // dst
        get_local_size(0) * sizeof(half), // src width
        get_local_size(0) * sizeof(half), // dst width
        get_local_size(0) * sizeof(half), // src stride
        get_global_size(0) * sizeof(half), // dst stride
        C, // num planes
        get_local_size(0) * get_local_size(1) * sizeof(half), // src plane stride
        get_global_size(0) * get_global_size(1) * sizeof(half), // dst plane stride
        get_local_size(0) * get_local_size(1) * sizeof(half), // plane size
        0);
}

__kernel void grn_NCHW(
    __global const half* restrict src,
    __global       half* restrict dst,
    __local        half* restrict local_src,
    __local        half* restrict local_dst,
    int C,
    float bias)
{
    float variance = bias + 1e-9f;

    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        float val = (float) local_src[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)];
        variance += val*val;
    }

    half hvariance = (half)(native_rsqrt((half)(variance/16.f))*0.25f);

    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        local_dst[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)]
        = local_src[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)] * hvariance;
    }
}

#else // defined (USE_MANUAL_DMA)

__kernel void grn_NCHW(
    __global const half* restrict src,
    __global       half* restrict dst,
    __local        half* restrict local_src, // unused, added for compatibility with DMA variant
    __local        half* restrict local_dst, // unused, added for compatibility with DMA variant
    int C,
    float bias)
{
    float variance = bias + 1e-9f;

    #pragma unroll 4
    for (int c = 0; c < C; c++)
    {
        float val = (float) src[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)];
        variance += val*val;
    }

    half hvariance = (half)(native_rsqrt((half)(variance/16.f))*0.25f);

    #pragma unroll 4
    for (int c = 0; c < C; c++)
    {
        dst[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)]
        = src[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)] * hvariance;
    }
}

#endif // defined (USE_MANUAL_DMA)
