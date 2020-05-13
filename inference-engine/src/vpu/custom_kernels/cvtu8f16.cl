// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define USE_MANUAL_DMA 1

#if defined (USE_MANUAL_DMA)

__kernel void __dma_preload_cvtu8f16(
    __global uchar* restrict src,
    __global half*  restrict dst,
    float   scale,
    float   bias,
    __local uchar* restrict local_src,
    __local half*  restrict local_dst)
{
    WorkGroupDmaCreate3DTransaction(
        src + get_group_id(0)*get_local_size(0)
            + get_group_id(1)*get_local_size(1)*get_global_size(0)
            + get_group_id(2)*get_local_size(2)*get_global_size(0)*get_global_size(1), // src
        local_src, // dst
        get_local_size(0) * sizeof(uchar), // src width
        get_local_size(0) * sizeof(uchar), // dst width
        get_global_size(0) * sizeof(uchar), // src stride
        get_local_size(0) * sizeof(uchar),  // dst stride
        get_local_size(2), // num planes
        get_global_size(0) * get_global_size(1) * sizeof(uchar), // src plane stride
        get_local_size(0) * get_local_size(1) * sizeof(uchar), // dst plane stride
        get_local_size(0) * get_local_size(1) * sizeof(uchar), // plane size
        0);
}

__kernel void __dma_postwrite_cvtu8f16(
    __global uchar* restrict src,
    __global half*  restrict dst,
    float   scale,
    float   bias,
    __local uchar* restrict local_src,
    __local half*  restrict local_dst)
{
    WorkGroupDmaCreate3DTransaction(
        local_dst, // src
        dst + get_group_id(0)*get_local_size(0)
            + get_group_id(1)*get_local_size(1)*get_global_size(0)
            + get_group_id(2)*get_local_size(2)*get_global_size(0)*get_global_size(1), // dst
        get_local_size(0) * sizeof(half), // src width
        get_local_size(0) * sizeof(half), // dst width
        get_local_size(0) * sizeof(half), // src stride
        get_global_size(0) * sizeof(half), // dst stride
        get_local_size(2), // num planes
        get_local_size(0) * get_local_size(1) * sizeof(half), // src plane stride
        get_global_size(0) * get_global_size(1) * sizeof(half), // dst plane stride
        get_local_size(0) * get_local_size(1) * sizeof(half), // plane size
        0);
}

__kernel void cvtu8f16(
    __global uchar* restrict src,
    __global half*  restrict dst,
    float   scale,
    float   bias,
    __local uchar* restrict local_src,
    __local half* restrict local_dst)
{
    size_t idx = get_local_id(0) +
                 get_local_id(1)*get_local_size(0) +
                 get_local_id(2)*get_local_size(0)*get_local_size(1);
    local_dst[idx] = convert_half(local_src[idx])*(half)scale+(half)bias;
}

#else // defined (USE_MANUAL_DMA)

__kernel void cvtu8f16(
    __global uchar* restrict src,
    __global half*  restrict dst,
    float   scale,
    float   bias,
    __local uchar* restrict local_src, // unused, added for compatibility with DMA variant
    __local half* restrict local_dst) // unused, added for compatibility with DMA variant
{
    int idx = get_global_id(0) +
              get_global_id(1) * get_global_size(0) +
              get_global_id(2) * get_global_size(0) * get_global_size(1);
    dst[idx] = convert_half(src[idx])*(half)scale+(half)bias;
}

#endif // defined (USE_MANUAL_DMA)

