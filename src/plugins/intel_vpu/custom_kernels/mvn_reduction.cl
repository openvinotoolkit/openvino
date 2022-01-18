// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

// Set to 1 only if output is zerroed before kernel execution
#define USE_ATOMICS 0

void atomic_add_global(volatile __global float *source, const float operand)
{
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
        newVal.floatVal  = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void reduction_mean(
    __global const half *restrict src,
    __global float *restrict mean,
    __global float *restrict variance,
    int W,
    int H,
    int across_channels)
{
    __local half src_line[4 * 1024];
    event_t e;

    e = async_work_group_copy_2D2D(
        src_line, // dst
        src + get_group_id(1) * get_local_size(1) * W
            + get_group_id(2) * get_local_size(2) * W * get_global_size(1), // src
        W * get_local_size(1), // num_elements_per_line,
        get_local_size(2), // num_lines,
        W * (get_global_size(1) - get_local_size(1)), // src_line_stride,
        0, // dst_line_stride,
        0);

    wait_group_events(1, &e);

    int h = get_global_id(1);
    int c = get_global_id(2);

    const int MAX_LOCAL_SIZE = 8;

    __local float mbuf[MAX_LOCAL_SIZE];
    __local float vbuf[MAX_LOCAL_SIZE];

    mbuf[get_local_id(1)] = 0;
    vbuf[get_local_id(1)] = 0;

    if (h < H) {
        float sum  = 0.f;
        float sum2 = 0.f;

        float8 sum4  = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float8 sum24 = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        const __local half8 *restrict lsrc = ((const __local half8 *)(src_line + get_local_id(1) * W));

        #pragma unroll 16
        for (size_t w = 0; w < W / 8; w++) {
            half8 sh    = lsrc[w];
            float8 valf = convert_float8(sh);

            sum4 += valf;
            sum24 += valf * valf;
        }

        for (size_t w = W / 8 * 8; w < W; w++) {
            float val = (float)src_line[get_local_id(1) * W + w];
            sum += val;
            sum2 += val * val;
        }

        mbuf[get_local_id(1)] = sum4.s0 + sum4.s1 + sum4.s2 + sum4.s3 + sum4.s4 + sum4.s5 + sum4.s6 + sum4.s7 + sum;
        vbuf[get_local_id(1)] =
            sum24.s0 + sum24.s1 + sum24.s2 + sum24.s3 + sum24.s4 + sum24.s5 + sum24.s6 + sum24.s7 + sum2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(1) == 0) {
        float res  = 0;
        float res2 = 0;

        for (int i = 0; i < get_local_size(1); i++) {
            res += mbuf[i];
            res2 += vbuf[i];
        }

// requires memory reset before layer execution
#if USE_ATOMICS
        int idx = (across_channels == 0) ? c : 0;

        atomic_add_global(mean + idx, res);
        atomic_add_global(variance + idx, res2);
#else
        int idx = c * get_num_groups(1) + get_group_id(1);

        mean[idx]     = res;
        variance[idx] = res2;
#endif
    }
}
