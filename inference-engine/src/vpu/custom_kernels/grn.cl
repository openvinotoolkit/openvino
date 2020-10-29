#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Define if runtime supports it. MX runtime is compatible, KMB is in WIP state
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
    __global const half* restrict src_data,
    __global       half* restrict dst_data,
    __local        half* restrict src,
    __local        half* restrict dst,
    int C,
    float bias)
{
    float variance = bias + 1e-9f;

    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        float val = (float) src[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)];
        variance += val*val;
    }

    half hvariance = (half)(native_rsqrt((half)(variance/16.f))*0.25f);

    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        dst[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)]
        = src[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)] * hvariance;
    }
}

#else // defined (USE_MANUAL_DMA)

__kernel void grn_NCHW(
    __global const half* restrict src_data,
    __global       half* restrict dst_data,
    __local        half* restrict src, // unused, added for compatibility with DMA variant
    __local        half* restrict dst, // unused, added for compatibility with DMA variant
    int C,
    float bias)
{
    float variance = bias + 1e-9f;

    #pragma unroll 4
    for (int c = 0; c < C; c++)
    {
        float val = (float) src_data[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)];
        variance += val*val;
    }

    half hvariance = (half)(native_rsqrt((half)(variance/16.f))*0.25f);

    #pragma unroll 4
    for (int c = 0; c < C; c++)
    {
        dst_data[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)]
        = src_data[c*get_global_size(1)*get_global_size(0) + get_global_id(1)*get_global_size(0) + get_global_id(0)] * hvariance;
    }
}

#endif // defined (USE_MANUAL_DMA)

// doesn't work right now due to compiler limitation
// ToDo: fix compiler
#if defined(IN_KERNEL_DMA)

#define MAX_LOCAL_W 224
#define MAX_LOCAL_H   2
#define MAX_LOCAL_C  24

__kernel void grn_NCHW(__global const half* restrict src_data,
                       __global       half* restrict dst_data,
                      int C,
                      float bias)
{
    __local half src[MAX_LOCAL_W*MAX_LOCAL_H*MAX_LOCAL_C]; // get_local_size(0)*get_local_size(1)*C
    __local half dst[MAX_LOCAL_W*MAX_LOCAL_H*MAX_LOCAL_C]; // get_local_size(0)*get_local_size(1)*C

    const size_t index = get_group_id(0)*get_local_size(0) + get_group_id(1)*get_local_size(1)*get_global_size(0);

    event_t e1 = async_work_group_copy_3D3D(
        src,                                                                            // dst
        src_data + index,                                                               // src
        get_local_size(0),                                                              // num_elements_per_line,
        get_local_size(1),                                                              // num_lines,
        get_global_size(0) - get_local_size(0),                                         // src_line_stride,
        0,                                                                              // dst_line_stride,
        C,                                                                              // num_planes,
        get_global_size(0)*get_global_size(1) - get_local_size(0) * get_local_size(1),  // src_plane_stride
        0,                                                                              // dst_plane_stride
        0);                                                                             // event
    wait_group_events(1, &e1);

    ////////////////////////

    float variance = bias + 1e-9f;

    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        float val = (float) src[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)];
        variance += val*val;
    }

    half hvariance = (half)(native_rsqrt((half)(variance/16.f))*0.25f);

    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        dst[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)]
        = src[c*get_local_size(1)*get_local_size(0) + get_local_id(1)*get_local_size(0) + get_local_id(0)] * hvariance;
    }

    ////////////////////

    event_t e2 = async_work_group_copy_3D3D(
        dst_data + index,                                                               // src
        dst,                                                                            // dst
        get_local_size(0),                                                              // num_elements_per_line,
        get_local_size(1),                                                              // num_lines,
        0,                                                                              // src_line_stride,
        get_global_size(0) - get_local_size(0),                                         // dst_line_stride,
        C,                                                                              // num_planes,
        0,                                                                              // src_plane_stride
        get_global_size(0)*get_global_size(1) - get_local_size(0) * get_local_size(1),  // dst_plane_stride
        0);                                                                             // event
    wait_group_events(1, &e2);
}
#endif
