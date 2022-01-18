// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define MAX_OPENCL_BUFF_SIZE 64 * 1024

#define USE_DMA 1

#if defined(USE_DMA)
void dmacpyLineSrcStrideStart(global half *from, private half *to, int size, int src_width, int src_stride)
{
    item_dma_event_t copyEvent =
        WorkItemDmaCreateStrideTransaction(from, to, src_width, src_width, src_stride, src_width, size, 0);
    WaitWorkItemDmaEvents(1, &copyEvent);
}

void dmacpyLineDstStrideStart(private half *from, global half *to, int size, int src_width, int src_stride)
{
    item_dma_event_t copyEvent =
        WorkItemDmaCreateStrideTransaction(from, to, src_width, src_width, src_width, src_stride, size, 0);
    WaitWorkItemDmaEvents(1, &copyEvent);
}
#endif

void memzero(void *ptr, size_t num)
{
    float4 *line0_ = (float4 *)ptr;
    #pragma unroll 16
    for (int i = 0; i < num / 16; i++) {
        line0_[i] = (float4){0.f, 0.f, 0.f, 0.f};
    }
    uchar *ptr_ = (uchar *)ptr;
    for (int i = num / 16 * 16; i < num; i++) {
        ptr_[i] = 0;
    }
}

void __attribute__((noinline)) crosscorrh(
    __private const half *restrict line0,
    __private const half *restrict line1,
    __private half *restrict dline,
    int topwidth,
    int max_displacement,
    int neighborhood_grid_radius,
    int kernel_size,
    int padding,
    int bottomwidth,
    int stride1,
    int stride2,
    int max_channels,
    int cur_subchannels)
{
    if (max_channels == 64) {
        for (int i = 0; i < kernel_size; i++) {
            int x1      = max_displacement - padding + i;
            int offset1 = x1 >= 0 ? 0 : (-x1 + stride1 - 1) / stride1;
            x1 += offset1 * stride1;

            for (int blockIdx_x = offset1; blockIdx_x < topwidth && x1 < bottomwidth; blockIdx_x++, x1 += stride1) {
                int x2      = x1 - neighborhood_grid_radius * stride2;
                int offset2 = x2 >= 0 ? 0 : (-x2 + stride2 - 1) / stride2;
                x2 += offset2 * stride2;

                for (int top_channel_x = offset2 - neighborhood_grid_radius;
                     top_channel_x <= neighborhood_grid_radius && x2 < bottomwidth;
                     top_channel_x++, x2 += stride2) {
                    half8 sum4 = (half8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

                    half8 *src0 = (half8 *)(line0 + x1 * max_channels);
                    half8 *src1 = (half8 *)(line1 + x2 * max_channels);

                    #pragma unroll 8
                    for (int ch = 0; ch < max_channels / 8; ch++) sum4 += (src0[ch]) * (src1[ch]);

                    half sum = __builtin_shave_sau_sumx_f16_r(sum4);
                    dline[(top_channel_x + neighborhood_grid_radius) * topwidth + blockIdx_x] += (sum);
                }
            }
        }
    } else {
        int neighborhood_grid_width = 2 * neighborhood_grid_radius + 1;

        for (int blockIdx_x = 0; blockIdx_x < topwidth; blockIdx_x++) {
            for (int i = 0; i < kernel_size; i++) {
                int x1 = blockIdx_x * stride1 + max_displacement + i - padding;

                if ((x1 >= 0) && (x1 < bottomwidth)) {
                    int o_min = -neighborhood_grid_radius * stride2;
                    int o_max = neighborhood_grid_width * stride2 - neighborhood_grid_radius * stride2;
                    if ((o_min) < (-x1)) {
                        o_min -= ((x1 + o_min - (stride2 - 1)) / stride2) * stride2;
                    }
                    if ((o_max) >= (bottomwidth + stride2 - x1)) {
                        o_max -= ((x1 + o_max - bottomwidth) / stride2) * stride2;
                    }

                    int o = o_min;
                    for (; o <= o_max - 4 * stride2; o += 4 * stride2) {
                        half8 *bottom0   = (half8 *)(line0 + x1 * max_channels);
                        half8 *bottom1_0 = (half8 *)(line1 + (x1 + o + 0 * stride2) * max_channels);
                        half8 *bottom1_1 = (half8 *)(line1 + (x1 + o + 1 * stride2) * max_channels);
                        half8 *bottom1_2 = (half8 *)(line1 + (x1 + o + 2 * stride2) * max_channels);
                        half8 *bottom1_3 = (half8 *)(line1 + (x1 + o + 3 * stride2) * max_channels);

                        int c = 0;

                        half8 sum40 = 0;
                        half8 sum41 = 0;
                        half8 sum42 = 0;
                        half8 sum43 = 0;

                        for (; c <= cur_subchannels / 8 - 4; c += 4) {
                            sum40 += bottom0[c + 0] * bottom1_0[c + 0];
                            sum40 += bottom0[c + 1] * bottom1_0[c + 1];
                            sum40 += bottom0[c + 2] * bottom1_0[c + 2];
                            sum40 += bottom0[c + 3] * bottom1_0[c + 3];

                            sum41 += bottom0[c + 0] * bottom1_1[c + 0];
                            sum41 += bottom0[c + 1] * bottom1_1[c + 1];
                            sum41 += bottom0[c + 2] * bottom1_1[c + 2];
                            sum41 += bottom0[c + 3] * bottom1_1[c + 3];

                            sum42 += bottom0[c + 0] * bottom1_2[c + 0];
                            sum42 += bottom0[c + 1] * bottom1_2[c + 1];
                            sum42 += bottom0[c + 2] * bottom1_2[c + 2];
                            sum42 += bottom0[c + 3] * bottom1_2[c + 3];

                            sum43 += bottom0[c + 0] * bottom1_3[c + 0];
                            sum43 += bottom0[c + 1] * bottom1_3[c + 1];
                            sum43 += bottom0[c + 2] * bottom1_3[c + 2];
                            sum43 += bottom0[c + 3] * bottom1_3[c + 3];
                        }

                        for (; c < cur_subchannels / 8; c++) {
                            sum40 += bottom0[c] * bottom1_0[c];
                            sum41 += bottom0[c] * bottom1_1[c];
                            sum42 += bottom0[c] * bottom1_2[c];
                            sum43 += bottom0[c] * bottom1_3[c];
                        }

                        half sum0 = __builtin_shave_sau_sumx_f16_r(sum40);
                        half sum1 = __builtin_shave_sau_sumx_f16_r(sum41);
                        half sum2 = __builtin_shave_sau_sumx_f16_r(sum42);
                        half sum3 = __builtin_shave_sau_sumx_f16_r(sum43);

                        for (c = c * 8; c < cur_subchannels; c++) {
                            sum0 += line0[x1 * max_channels + c] * line1[(x1 + o + 0 * stride2) * max_channels + c];
                            sum1 += line0[x1 * max_channels + c] * line1[(x1 + o + 1 * stride2) * max_channels + c];
                            sum2 += line0[x1 * max_channels + c] * line1[(x1 + o + 2 * stride2) * max_channels + c];
                            sum3 += line0[x1 * max_channels + c] * line1[(x1 + o + 3 * stride2) * max_channels + c];
                        }

                        dline[blockIdx_x + (((o / stride2) + 0) * topwidth + neighborhood_grid_radius * topwidth)] +=
                            sum0;
                        dline[blockIdx_x + (((o / stride2) + 1) * topwidth + neighborhood_grid_radius * topwidth)] +=
                            sum1;
                        dline[blockIdx_x + (((o / stride2) + 2) * topwidth + neighborhood_grid_radius * topwidth)] +=
                            sum2;
                        dline[blockIdx_x + (((o / stride2) + 3) * topwidth + neighborhood_grid_radius * topwidth)] +=
                            sum3;
                    }

                    for (; o < o_max; o += 1 * stride2) {
                        half8 *bottom0 = (half8 *)(line0 + x1 * max_channels);
                        half8 *bottom1 = (half8 *)(line1 + (x1 + o) * max_channels);

                        int c = 0;

                        half8 sum4 = 0;
                        for (; c <= cur_subchannels / 8 - 4; c += 4) {
                            sum4 += bottom0[c + 0] * bottom1[c + 0];
                            sum4 += bottom0[c + 1] * bottom1[c + 1];
                            sum4 += bottom0[c + 2] * bottom1[c + 2];
                            sum4 += bottom0[c + 3] * bottom1[c + 3];
                        }
                        for (; c < cur_subchannels / 8; c++) {
                            sum4 += bottom0[c] * bottom1[c];
                        }

                        half sum = __builtin_shave_sau_sumx_f16_r(sum4);

                        for (c = c * 8; c < cur_subchannels; c++) {
                            sum += line0[x1 * max_channels + c] * line1[(x1 + o) * max_channels + c];
                        }

                        dline[blockIdx_x + (((o + neighborhood_grid_radius * stride2) / stride2) * topwidth)] += sum;
                    }
                }
            }
        }
    }
}

__kernel void correlate2_half(
    __global const half *restrict bottom0,
    __global const half *restrict bottom1,
    __global half *restrict top,
    int topwidth,
    int topheight,
    int bottomwidth,
    int bottomheight,
    int bottomchannels,
    int max_displacement,
    int padding,
    int neighborhood_grid_radius,
    int neighborhood_grid_width,
    int kernel_size,
    int stride1,
    int stride2)
{
    int max_channels = (MAX_OPENCL_BUFF_SIZE / sizeof(half) - topwidth * neighborhood_grid_width) / (3 * bottomwidth);
    if (max_channels > 64) max_channels = 64;
    int subchannels_count = (bottomchannels + max_channels - 1) / max_channels;
    int subchannels       = (bottomchannels + subchannels_count - 1) / subchannels_count;
    if (subchannels < max_channels) subchannels = max_channels;

    const int sumelems = kernel_size * kernel_size * bottomchannels;

    __private half cmx[MAX_OPENCL_BUFF_SIZE / sizeof(half)];

    __private half *line0 = cmx;
    __private half *line1 = line0 + bottomwidth * subchannels;
    __private half *dline = line1 + bottomwidth * subchannels;

    int blockIdx_y = get_global_id(0);

#if defined(USE_DMA)
    __private half *dmabuf = dline + topwidth * neighborhood_grid_width;
#endif

    int y1 = blockIdx_y * stride1 + max_displacement;

    for (int j = 0; j < kernel_size; j++) {
        for (int bottomchannel = 0; bottomchannel < bottomchannels; bottomchannel += subchannels) {
            // configure channel batching
            int startchannel = bottomchannel;
            int endchannel = startchannel + subchannels > bottomchannels ? bottomchannels : startchannel + subchannels;
            int deltachannels = endchannel - startchannel;

            // load line form blob 0 with repackaging
            if (y1 + j - padding >= 0 && y1 + j - padding < bottomheight) {
#if defined(USE_DMA)
                __global const half *curr =
                    bottom0 + startchannel * bottomheight * bottomwidth + (y1 + j - padding) * bottomwidth;
                dmacpyLineSrcStrideStart(
                    curr,
                    dmabuf,
                    bottomwidth * deltachannels * sizeof(half),
                    bottomwidth * sizeof(half),
                    bottomwidth * bottomheight * sizeof(half));

                for (int ch = 0; ch < deltachannels; ch++) {
                    for (int blockIdx_x = 0; blockIdx_x < bottomwidth / 8; blockIdx_x++) {
                        half8 val = ((half8 *)(dmabuf + ch * bottomwidth))[blockIdx_x];
                        line0[(blockIdx_x * 8 + 0) * max_channels + ch] = val[0];
                        line0[(blockIdx_x * 8 + 1) * max_channels + ch] = val[1];
                        line0[(blockIdx_x * 8 + 2) * max_channels + ch] = val[2];
                        line0[(blockIdx_x * 8 + 3) * max_channels + ch] = val[3];

                        line0[(blockIdx_x * 8 + 4) * max_channels + ch] = val[4];
                        line0[(blockIdx_x * 8 + 5) * max_channels + ch] = val[5];
                        line0[(blockIdx_x * 8 + 6) * max_channels + ch] = val[6];
                        line0[(blockIdx_x * 8 + 7) * max_channels + ch] = val[7];
                    }

                    for (int blockIdx_x = bottomwidth / 8 * 8; blockIdx_x < bottomwidth; blockIdx_x++) {
                        line0[(blockIdx_x)*max_channels + ch] = dmabuf[blockIdx_x + ch * bottomwidth];
                    }
                }

                if (deltachannels < subchannels)
                    for (int blockIdx_x = 0; blockIdx_x < bottomwidth; blockIdx_x++)
                        memzero(
                            line0 + blockIdx_x * max_channels + deltachannels,
                            (subchannels - deltachannels) * sizeof(half));
#else
                for (int blockIdx_x = 0; blockIdx_x < bottomwidth; blockIdx_x++) {
                    for (int ch = 0; ch < deltachannels; ch++)
                        line0[blockIdx_x * max_channels + ch] = bottom0
                            [(ch + startchannel) * bottomheight * bottomwidth + (y1 + j - padding) * bottomwidth
                             + blockIdx_x];

                    if (deltachannels < subchannels)
                        memzero(
                            line0 + blockIdx_x * max_channels + deltachannels,
                            (subchannels - deltachannels) * sizeof(half));
                }
#endif
            } else
                memzero(line0, max_channels * bottomwidth * sizeof(half));

            for (int top_channel_y = 0; top_channel_y < neighborhood_grid_width; top_channel_y++) {
                int y2 = y1 + (top_channel_y - neighborhood_grid_radius) * stride2;

                if (y2 + j - padding >= 0 && y2 + j - padding < bottomheight) {
#if defined(USE_DMA)
                    __global const half *curr =
                        bottom1 + startchannel * bottomheight * bottomwidth + (y2 + j - padding) * bottomwidth;
                    dmacpyLineSrcStrideStart(
                        curr,
                        dmabuf,
                        bottomwidth * deltachannels * sizeof(half),
                        bottomwidth * sizeof(half),
                        bottomwidth * bottomheight * sizeof(half));

                    for (int ch = 0; ch < deltachannels; ch++) {
                        for (int blockIdx_x = 0; blockIdx_x < bottomwidth / 8; blockIdx_x++) {
                            half8 val = ((half8 *)(dmabuf + ch * bottomwidth))[blockIdx_x];
                            line1[(blockIdx_x * 8 + 0) * max_channels + ch] = val[0];
                            line1[(blockIdx_x * 8 + 1) * max_channels + ch] = val[1];
                            line1[(blockIdx_x * 8 + 2) * max_channels + ch] = val[2];
                            line1[(blockIdx_x * 8 + 3) * max_channels + ch] = val[3];

                            line1[(blockIdx_x * 8 + 4) * max_channels + ch] = val[4];
                            line1[(blockIdx_x * 8 + 5) * max_channels + ch] = val[5];
                            line1[(blockIdx_x * 8 + 6) * max_channels + ch] = val[6];
                            line1[(blockIdx_x * 8 + 7) * max_channels + ch] = val[7];
                        }

                        for (int blockIdx_x = bottomwidth / 8 * 8; blockIdx_x < bottomwidth; blockIdx_x++) {
                            line1[(blockIdx_x)*max_channels + ch] = dmabuf[blockIdx_x + ch * bottomwidth];
                        }
                    }
#else
                    for (int ch = 0; ch < deltachannels; ch++) {
                        for (int blockIdx_x = 0; blockIdx_x < bottomwidth / 8; blockIdx_x++) {
                            half8 val = ((
                                __global half8
                                    *)(bottom1 + (ch + startchannel) * bottomheight * bottomwidth + (y2 + j - padding) * bottomwidth))
                                [blockIdx_x];
                            line1[(blockIdx_x * 8 + 0) * max_channels + ch] = val[0];
                            line1[(blockIdx_x * 8 + 1) * max_channels + ch] = val[1];
                            line1[(blockIdx_x * 8 + 2) * max_channels + ch] = val[2];
                            line1[(blockIdx_x * 8 + 3) * max_channels + ch] = val[3];

                            line1[(blockIdx_x * 8 + 4) * max_channels + ch] = val[4];
                            line1[(blockIdx_x * 8 + 5) * max_channels + ch] = val[5];
                            line1[(blockIdx_x * 8 + 6) * max_channels + ch] = val[6];
                            line1[(blockIdx_x * 8 + 7) * max_channels + ch] = val[7];
                        }
                        for (int blockIdx_x = bottomwidth / 8 * 8; blockIdx_x < bottomwidth; blockIdx_x++) {
                            half val =
                                (bottom1 + (ch + startchannel) * bottomheight * bottomwidth
                                 + (y2 + j - padding) * bottomwidth)[blockIdx_x];
                            line1[(blockIdx_x)*max_channels + ch] = val;
                        }
                    }
#endif
                    for (int blockIdx_x = 0; blockIdx_x < bottomwidth; blockIdx_x++) {
                        if (deltachannels < subchannels)
                            memzero(
                                line1 + blockIdx_x * max_channels + deltachannels,
                                (subchannels - deltachannels) * sizeof(half));
                    }
                } else
                    memzero(line1, max_channels * bottomwidth * sizeof(half));

                if (j == 0 && startchannel == 0) {
                    memzero(dline, neighborhood_grid_width * topwidth * sizeof(half));
                } else {
#if defined(USE_DMA)
                    dmacpyLineSrcStrideStart(
                        top + top_channel_y * neighborhood_grid_width * topheight * topwidth + blockIdx_y * topwidth,
                        dline,
                        topwidth * neighborhood_grid_width * sizeof(half),
                        topwidth * sizeof(half),
                        topwidth * topheight * sizeof(half));
#else
                    for (int top_channel_x = 0; top_channel_x < neighborhood_grid_width; top_channel_x++) {
                        for (int blockIdx_x = 0; blockIdx_x < topwidth / 8; blockIdx_x++) {
                            half8 val = ((
                                __global half8
                                    *)(top + ((top_channel_y * neighborhood_grid_width + top_channel_x) * topheight * topwidth + blockIdx_y * topwidth)))
                                [blockIdx_x];
                            ((half8 *)(dline + top_channel_x * topwidth))[blockIdx_x] = val;
                        }
                        for (int blockIdx_x = (topwidth / 8) * 8; blockIdx_x < topwidth; blockIdx_x++) {
                            dline[top_channel_x * topwidth + blockIdx_x] =
                                top[(top_channel_y * neighborhood_grid_width + top_channel_x) * topheight * topwidth
                                    + blockIdx_y * topwidth + blockIdx_x];
                        }
                    }
#endif
                }

                if (y1 + j - padding >= 0 && y1 + j - padding < bottomheight && y2 + j - padding >= 0
                    && y2 + j - padding < bottomheight) {
                    crosscorrh(
                        line0,
                        line1,
                        dline,
                        topwidth,
                        max_displacement,
                        neighborhood_grid_radius,
                        kernel_size,
                        padding,
                        bottomwidth,
                        stride1,
                        stride2,
                        max_channels,
                        subchannels);
                }

                if (j == kernel_size - 1 && endchannel == bottomchannels) {
                    half8 scale = (half8){
                        (half)sumelems,
                        (half)sumelems,
                        (half)sumelems,
                        (half)sumelems,
                        (half)sumelems,
                        (half)sumelems,
                        (half)sumelems,
                        (half)sumelems};
                    for (int top_channel_x = 0; top_channel_x < neighborhood_grid_width; top_channel_x++) {
                        for (int blockIdx_x = 0; blockIdx_x < topwidth / 8; blockIdx_x++) {
                            ((half8 *)(dline + top_channel_x * topwidth))[blockIdx_x] =
                                ((half8 *)(dline + top_channel_x * topwidth))[blockIdx_x] / scale;
                        }
                        for (int blockIdx_x = (topwidth / 8) * 8; blockIdx_x < topwidth; blockIdx_x++) {
                            dline[top_channel_x * topwidth + blockIdx_x] =
                                dline[top_channel_x * topwidth + blockIdx_x] / (half)sumelems;
                        }
                    }
                }

#if defined(USE_DMA)
                dmacpyLineDstStrideStart(
                    dline,
                    top + top_channel_y * neighborhood_grid_width * topheight * topwidth + blockIdx_y * topwidth,
                    topwidth * neighborhood_grid_width * sizeof(half),
                    topwidth * sizeof(half),
                    topwidth * topheight * sizeof(half));
#else
                for (int top_channel_x = 0; top_channel_x < neighborhood_grid_width; top_channel_x++) {
                    for (int blockIdx_x = 0; blockIdx_x < topwidth / 8; blockIdx_x++) {
                        ((__global half8
                              *)(top + ((top_channel_y * neighborhood_grid_width + top_channel_x) * topheight * topwidth + blockIdx_y * topwidth)))
                            [blockIdx_x] = ((half8 *)(dline + top_channel_x * topwidth))[blockIdx_x]
                                           + (half8){0, 0, 0, 0, 0, 0, 0, 0};
                    }
                    for (int blockIdx_x = (topwidth / 8) * 8; blockIdx_x < topwidth; blockIdx_x++) {
                        top[(top_channel_y * neighborhood_grid_width + top_channel_x) * topheight * topwidth
                            + blockIdx_y * topwidth + blockIdx_x] =
                            dline[top_channel_x * topwidth + blockIdx_x] + (half)0;
                    }
                }
#endif
            }
        }
    }
}
