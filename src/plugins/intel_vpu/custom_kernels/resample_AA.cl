// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_extended_async_copies : enable

#define USE_OPTIMIZED_ROUND

#ifdef USE_OPTIMIZED_ROUND
#define ROUND(x) ((int)((x) + 0.5f))
#else
#define ROUND(x) (int)(round(x))
#endif

inline int out_to_in(float ox, float f)
{
#ifdef USE_OPTIMIZED_ROUND
    return (int)((ox + 0.5f) / f);
#else
    return ROUND((ox + 0.5f) / f - 0.5f);
#endif
}

static inline float triangleCoeff(float x) { return 1.0f - fabs(x); }

static inline float4 triangleCoeff4(float4 x) { return 1.0f - fabs(x); }

__kernel void resample_with_antialias(
    __global const half *restrict src,
    __global half *restrict dst,
    int iw,
    int ih,
    float factor,
    int ow,
    int oh,
    int channels)
{
    __local half local_src[20 * 1024];
    __local half local_dst[8 * 1024];

    const int r        = (factor > 1.0f) ? 2 : ceil(1.0f / factor);
    const int oy_first = get_group_id(1) * get_local_size(1);
    const int oy_last  = (get_group_id(1) + 1) * get_local_size(1) - 1;
    const int iy_first = max(out_to_in(oy_first, factor) - r, 0);
    const int iy_last  = min(out_to_in(oy_last, factor) + r, ih - 1);
    const int iy_size  = iy_last - iy_first + 1;

    event_t e1 = async_work_group_copy_2D2D(
        local_src, // dst
        src + get_group_id(2) * get_local_size(2) * ih * iw + iy_first * iw, // src
        iy_size * iw, // num_elements_per_line,
        get_local_size(2), // num_lines,
        (ih - iy_size) * iw, // src_line_stride,
        0, // dst_line_stride,
        0);
    wait_group_events(1, &e1);

    const int oy     = get_global_id(1);
    const float iy_f = ((oy + 0.5f) / factor - 0.5f) - iy_first;
    const int iy     = ROUND(iy_f);

    __local half const *restrict start_src =
        local_src + iw * get_local_id(1) + iw * iy_size * get_local_id(2);
    __local half *restrict start_dst =
        local_dst + ow * get_local_id(1) + ow * get_local_size(1) * get_local_id(2);

    for (int ox = 0; ox < ow; ox++) {
        const float ix_f = (float)((ox + 0.5f) / factor) - 0.5f;
        const int ix_i   = ROUND(ix_f);

        float4 v_sum  = 0.f;
        float4 v_wsum = 0.f;
        for (int y = 0; y < iy_size; y++) {
            float dy  = iy_f - y;
            int x     = max(ix_i - r, 0);
            int end_x = min(ix_i + r, iw - 1);

            float4 dx;
            for (int i = 0; i < 4; i++) dx[i] = ix_f - x - i;

            for (; x < end_x - 3; x += 4, dx -= 4) {
                float4 w =
                    factor * triangleCoeff4(factor * dx) * factor * triangleCoeff(factor * dy);
                float4 src_vec = {
                    start_src[y * iw + x + 0],
                    start_src[y * iw + x + 1],
                    start_src[y * iw + x + 2],
                    start_src[y * iw + x + 3]};

                v_sum += w * src_vec;
                v_wsum += w;
            }

            for (; x <= end_x; x++) {
                float dx = ix_f - x;
                float w = factor * triangleCoeff(factor * dx) * factor * triangleCoeff(factor * dy);

                v_sum[0] += w * start_src[y * iw + x];
                v_wsum[0] += w;
            }
        }

        v_sum[0]  = v_sum[0] + v_sum[1] + v_sum[2] + v_sum[3];
        v_wsum[0] = v_wsum[0] + v_wsum[1] + v_wsum[2] + v_wsum[3];

        start_dst[get_local_id(1) * ow + ox] = (!v_wsum[0]) ? 0.0f : (half)(v_sum[0] / v_wsum[0]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    event_t e2 = async_work_group_copy_2D2D(
        dst + get_group_id(2) * get_local_size(2) * get_global_size(1) * ow
            + get_group_id(1) * get_local_size(1) * ow, // dst
        local_dst, // src
        get_local_size(1) * ow, // num_elements_per_line,
        get_local_size(2), // num_lines,
        0, // src_line_stride,
        (get_global_size(1) - get_local_size(1)) * ow, // dst_line_stride,
        0);
    wait_group_events(1, &e2);
}
