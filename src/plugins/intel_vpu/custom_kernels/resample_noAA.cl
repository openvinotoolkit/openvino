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

inline int out_to_in(float ox, float f) { return (int)((ox + 0.5f) * f); }

void interpolationCHW_nn(__local half *psrc, __local half *pdst, int OW, int IW, int C, float rw, float rh)
{
    float alpha = rh / 2.0f - 0.5f;

    for (int w = 0; w < OW / 8; w++) {
        float fw0 = rw * (w * 8 + 0) + alpha;
        float fw1 = rw * (w * 8 + 1) + alpha;
        float fw2 = rw * (w * 8 + 2) + alpha;
        float fw3 = rw * (w * 8 + 3) + alpha;

        float fw4 = rw * (w * 8 + 4) + alpha;
        float fw5 = rw * (w * 8 + 5) + alpha;
        float fw6 = rw * (w * 8 + 6) + alpha;
        float fw7 = rw * (w * 8 + 7) + alpha;

        int iw0 = min((int)ROUND(fw0), IW - 1);
        int iw1 = min((int)ROUND(fw1), IW - 1);
        int iw2 = min((int)ROUND(fw2), IW - 1);
        int iw3 = min((int)ROUND(fw3), IW - 1);

        int iw4 = min((int)ROUND(fw4), IW - 1);
        int iw5 = min((int)ROUND(fw5), IW - 1);
        int iw6 = min((int)ROUND(fw6), IW - 1);
        int iw7 = min((int)ROUND(fw7), IW - 1);

        for (int c = 0; c < C; c++) {
            half8 val = {
                *((__local half *)(psrc + c * IW + iw0)),
                *((__local half *)(psrc + c * IW + iw1)),
                *((__local half *)(psrc + c * IW + iw2)),
                *((__local half *)(psrc + c * IW + iw3)),

                *((__local half *)(psrc + c * IW + iw4)),
                *((__local half *)(psrc + c * IW + iw5)),
                *((__local half *)(psrc + c * IW + iw6)),
                *((__local half *)(psrc + c * IW + iw7)),
            };
            *((__local half8 *)(pdst + c * OW + w * 8)) = val;
        }
    }

    for (int w = OW / 8 * 8; w < OW; w++) {
        float fw = rw * w + alpha;
        int iw0  = min((int)ROUND(fw), IW - 1);

        for (int c = 0; c < C; c++) {
            *((__local half *)(pdst + c * OW + w)) = *((__local half *)(psrc + c * IW + iw0));
        }
    }
}

kernel void resample_nearest(
    __global const half *restrict src,
    __global half *restrict dst,
    int iw,
    int ih,
    float factor,
    int ow,
    int oh,
    int channels)
{
    __local half local_src[14 * 1024];
    __local half local_dst[14 * 1024];

    const int oy_first = get_group_id(1) * get_local_size(1);
    const int oy_last  = (get_group_id(1) + 1) * get_local_size(1) - 1;
    const int iy_first = out_to_in(oy_first, 1.0 / factor);
    const int iy_last  = out_to_in(oy_last, 1.0 / factor);

    const int iy_size = iy_last - iy_first + 1;

    event_t e1 = async_work_group_copy_2D2D(
        local_src, // dst
        src + get_group_id(2) * channels * ih * iw + iy_first * iw, // src
        iy_size * iw, // num_elements_per_line,
        channels, // num_lines,
        ih * iw - iy_size * iw, // src_line_stride,
        0, // dst_line_stride,
        0);

    wait_group_events(1, &e1);

    interpolationCHW_nn(local_src, local_dst, ow, iw, channels, 1.0 / factor, 1.0 / factor);

    event_t e2 = async_work_group_copy_2D2D(
        dst + get_group_id(2) * channels * get_global_size(1) * ow + get_group_id(1) * get_local_size(1) * ow, // dst
        local_dst, // src
        get_local_size(1) * ow, // size_t num_elements_per_line,
        channels, // size_t num_lines,
        0, // size_t src_line_stride,
        get_global_size(1) * ow - get_local_size(1) * ow, // size_t dst_line_stride,
        0);

    wait_group_events(1, &e2);
}
