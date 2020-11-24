// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_layers_tests.hpp"
#include <vector>
#include <array>
#include <algorithm>

using namespace InferenceEngine;

static void refShuffleChannel(const Blob::Ptr src,
                              Blob::Ptr dst,
                              int group, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
          uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);

    int G = group;
    int CX = IC / G;
    int CY = G;

    for (int cy = 0; cy < CY; cy++) {
        for (int cx = 0; cx < CX; cx++) {
            for (int h = 0; h < IH; h++) {
                for (int w = 0; w < IW; w++) {
                    if (isCHW) {
                        dst_data[(cx*CY + cy)*IW*IH + h*IW + w] = src_data[(cy*CX + cx)*IW*IH + h*IW + w];
                    } else {
                        dst_data[(cx*CY + cy) + h*IW*IC + w*IC] = src_data[(cy*CX + cx) + h*IW*IC + w*IC];
                    }
                }
            }
        }
    }
}

static void refQuantize(const Blob::Ptr src,
                        const Blob::Ptr input_low,
                        const Blob::Ptr input_high,
                        const Blob::Ptr output_low,
                        const Blob::Ptr output_high,
                        Blob::Ptr dst,
                        int levels, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(input_low, nullptr);
    ASSERT_NE(input_high, nullptr);
    ASSERT_NE(output_low, nullptr);
    ASSERT_NE(output_high, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    const uint16_t *input_low_data = input_low->buffer();
    const uint16_t *input_high_data = input_high->buffer();
    const uint16_t *output_low_data = output_low->buffer();
    const uint16_t *output_high_data = output_high->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(input_low_data, nullptr);
    ASSERT_NE(input_high_data, nullptr);
    ASSERT_NE(output_low_data, nullptr);
    ASSERT_NE(output_high_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t W = 1;
    int32_t H = 1;
    int32_t C = 1;

    get_dims(src, W, H, C);

    for (int c = 0; c < C; c++) {
        float ilow  = PrecisionUtils::f16tof32(input_low->size()   == 1 ? input_low_data[0]   : input_low_data[c]);
        float ihigh = PrecisionUtils::f16tof32(input_high->size()  == 1 ? input_high_data[0]  : input_high_data[c]);
        float olow  = PrecisionUtils::f16tof32(output_low->size()  == 1 ? output_low_data[0]  : output_low_data[c]);
        float ohigh = PrecisionUtils::f16tof32(output_high->size() == 1 ? output_high_data[0] : output_high_data[c]);

        // emulate half math to be close to half float SHAVE implementation
		float a = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((float)(levels - 1) / (ihigh - ilow)));
		float b = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((ohigh - olow) / (float)(levels - 1)));

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = (isCHW) ? c*W*H + h*W + w : c + h*W*C + w*C;
                float src_val = PrecisionUtils::f16tof32(src_data[idx]);
                float dst_val;

                if (src_val <= ilow) {
                    dst_val = olow;
                } else if (src_val > ihigh) {
                    dst_val = ohigh;
				} else {
                	if(!(ihigh - ilow) || !(levels - 1)) {
						dst_val = olow;
					} else {
						// quantization pass
						float quantized = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((src_val - ilow) * a));
						// de-quantization pass
						dst_val = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16(roundf(quantized) * b)) + olow;
					}
                }

                dst_data[idx] = PrecisionUtils::f32tof16(dst_val);
            }
        }
    }
}

static void ref_QuantizeBinarization(const Blob::Ptr src,
                        const Blob::Ptr input_low,
                        const Blob::Ptr input_high,
                        const Blob::Ptr output_low,
                        const Blob::Ptr output_high,
                        Blob::Ptr dst,
                        int levels) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(input_low, nullptr);
    ASSERT_NE(input_high, nullptr);
    ASSERT_NE(output_low, nullptr);
    ASSERT_NE(output_high, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    const uint16_t *input_low_data = input_low->buffer();
    const uint16_t *input_high_data = input_high->buffer();
    const uint16_t *output_low_data = output_low->buffer();
    const uint16_t *output_high_data = output_high->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(input_low_data, nullptr);
    ASSERT_NE(input_high_data, nullptr);
    ASSERT_NE(output_low_data, nullptr);
    ASSERT_NE(output_high_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t W = 1;
    int32_t H = 1;
    int32_t C = 1;
    get_dims(src, W, H, C);

    for (int c = 0; c < C; c++) {
        float ilow  = PrecisionUtils::f16tof32(input_low->size()   == 1 ? input_low_data[0]   : input_low_data[c]);
        float ihigh = PrecisionUtils::f16tof32(input_high->size()  == 1 ? input_high_data[0]  : input_high_data[c]);
        float olow  = PrecisionUtils::f16tof32(output_low->size()  == 1 ? output_low_data[0]  : output_low_data[c]);
        float ohigh = PrecisionUtils::f16tof32(output_high->size() == 1 ? output_high_data[0] : output_high_data[c]);

        // emulate half math to be close to half float SHAVE implementation
        float hTof_ilow = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16(ilow));
        float hTof_ihigh = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16(ihigh));
        float a = (0.01 > (hTof_ihigh - hTof_ilow)) ? 0.0f : PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((float)(levels - 1) / (hTof_ihigh - hTof_ilow)));
        float b = !(levels - 1) ? 0.0f : PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((ohigh - olow) / (float)(levels - 1)));

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = c*W*H + h*W + w;
                float src_val = PrecisionUtils::f16tof32(src_data[idx]);
                float dst_val;

                if (src_val <= ilow) {
                    dst_val = olow;
                } else if (src_val > ihigh) {
                    dst_val = ohigh;
                } else {
                    if(!(ihigh - ilow) || !(levels - 1))
                        dst_val = olow;
                    else
                    {
                        // quantization pass
                        float quantized = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16((src_val - ilow) * a));
                        // de-quantization pass
                        dst_val = PrecisionUtils::f16tof32(PrecisionUtils::f32tof16(roundf( quantized ) * b)) + olow;
                    }
                }

                dst_data[idx] = PrecisionUtils::f32tof16(dst_val);
            }
        }
    }
}

static void refBinaryConvolution(const Blob::Ptr src, const Blob::Ptr weights, Blob::Ptr dst,
                                 int dilations, int group, param_size kernel, int strides,
                                 int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t* src_data = src->buffer();
    const uint8_t*  weights_data = weights->buffer();
          uint16_t* dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(weights_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    get_dims(dst, OW, OH, OC);

    int KW = kernel.x;
    int KH = kernel.y;
    int KD = 1;

    int SW = strides;
    int SH = strides;
    int SD = 0;

    int DW = dilations;
    int DH = dilations;
    int DD = 0;

    int PW = kernel.x/2;
    int PH = kernel.y/2;
    int PD = 0;

    int GC = group;

    int ID = 1;
    int OD = 1;

    int pad_value = 0;

    int nbits = 8;

    auto extract_weights = [](uint8_t val, uint8_t bit) -> int {
        return (uint8_t)((val >> bit) & 1);
    };

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t od = 0; od < OD; od++) {
                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        int oidx = (isCHW) ? g  * OC / GC * OD * OH * OW +
                                             oc *           OD * OH * OW +
                                             od *           OH * OW +
                                             oh *           OW +
                                             ow
                                           : g  * OC / GC * OD +
                                             oc * OD +
                                             od +
                                             oh * OW * OC +
                                             ow * OC;

                        int dst_val = 0;

                        for (int ic = 0; ic < IC / GC; ic++) {
                            for (int kd = 0; kd < KD; kd++) {
                                for (int kh = 0; kh < KH; kh++) {
                                    for (int kw = 0; kw < KW; kw++) {
                                        int widx = g  * OC / GC * IC / GC * KD * KH * KW +
                                                   oc * IC / GC * KD * KH * KW +
                                                   ic * KD * KH * KW +
                                                   kd * KH * KW +
                                                   kh * KW +
                                                   kw;
                                        int w = extract_weights(weights_data[widx/nbits], (uint8_t)(widx % nbits));

                                        int s;

                                        int iw = ow * SW - PW + kw * DW;
                                        int ih = oh * SH - PH + kh * DH;
                                        int id = od * SD - PD + kd * DD;
                                        if (iw < 0 || iw >= (int) IW ||
                                            ih < 0 || ih >= (int) IH ||
                                            id < 0 || id >= (int) ID) {
                                            s = pad_value;
                                        } else {
                                            int iidx = (isCHW) ? g  * IC / GC * ID * IH * IW +
                                                                 ic * ID * IH * IW +
                                                                 id * IH * IW +
                                                                 ih * IW +
                                                                 iw
                                                               : g  * IC / GC * ID +
                                                                 ic * ID +
                                                                 id +
                                                                 ih * IW * IC +
                                                                 iw * IC;
                                            s = ((PrecisionUtils::f16tof32(src_data[iidx]) > 0.f) ? 1 : 0);
                                        }

                                        dst_val += s ^ w;
                                    }
                                }
                            }
                        }

                        dst_data[oidx] = PrecisionUtils::f32tof16((float)(IC/GC*KD*KH*KW - 2*dst_val));
                    }
                }
            }
        }
    }
}

static void refExperimentalDetectronPriorGridGenerator(
        std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs,
        int grid_h, int grid_w, int stride_h, int stride_w) {
    int num_priors = inputs[0]->getTensorDesc().getDims()[0];

    uint16_t *src_data = inputs[0]->buffer();
    uint16_t *dst_data = outputs[0]->buffer();

    using namespace PrecisionUtils;

    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            for (int s = 0; s < 3; ++s) {
                dst_data[0] = f32tof16(
                        f16tof32(src_data[4 * s + 0]) + stride_w * (w + 0.5f));
                dst_data[1] = f32tof16(
                        f16tof32(src_data[4 * s + 1]) + stride_h * (h + 0.5f));
                dst_data[2] = f32tof16(
                        f16tof32(src_data[4 * s + 2]) + stride_w * (w + 0.5f));
                dst_data[3] = f32tof16(
                        f16tof32(src_data[4 * s + 3]) + stride_h * (h + 0.5f));
                dst_data += 4;
            }
        }
    }
}

static void rearrange(const ie_fp16* in, ie_fp16* out, int num, int channels, int width, int height,
               int widthheight, int padding, int pwidthheight)
{
    (void) height;
    (void) pwidthheight;

    ASSERT_TRUE(num == 1) << "batch is not supported for Myriad";

    for (int xy = 0; xy < widthheight; xy++)
    {
        for (int ch = 0; ch < channels; ch++)
        {
            ie_fp16 value = in[ch * widthheight + xy];

            int xpad  = (xy % width + padding);
            int ypad  = (xy / width + padding);
            int xypad = ypad * (width + 2 * padding) + xpad;

            out[xypad * channels + ch] = value;
        }
    }
}

static void correlate(int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
                      int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width,
                      int kernel_radius, int kernel_size, int stride1, int stride2,
                      int bottomwidth, int bottomheight, int bottomchannels,
                      const ie_fp16* bottom0, const ie_fp16* bottom1, ie_fp16* top)
{
    (void) nthreads;
    (void) kernel_radius;
    (void) topcount;
    (void) bottomheight;
    (void) num;

    const int sumelems = kernel_size * kernel_size * bottomchannels;

    auto patch_data = std::vector<ie_fp16>(sumelems);

    for (int blockIdx_y = 0; blockIdx_y < topheight; blockIdx_y++)
    {
        for (int blockIdx_x = 0; blockIdx_x < topwidth; blockIdx_x++)
        {
            int x1 = blockIdx_x * stride1 + max_displacement;
            int y1 = blockIdx_y * stride1 + max_displacement;
            // Load 3D patch into shared memory
            for (int j = 0; j < kernel_size; j++)
            {
                for (int i = 0; i < kernel_size; i++)
                {
                    int idx1 = (      j  * kernel_size      + i) * bottomchannels;
                    int idx2 = ((y1 + j) * bottomwidth + x1 + i) * bottomchannels;

                    for (int ch = 0; ch < bottomchannels; ch++)
                        patch_data[idx1 + ch] = bottom0[idx2 + ch];
                }
            }

            for (int top_channel = 0; top_channel < topchannels; top_channel++)
            {
                int x2 = x1 + (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
                int y2 = y1 + (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;

                float sum = (0.0f);
                for (int j = 0; j < kernel_size; j++)
                {
                    for (int i = 0; i < kernel_size; i++)
                    {
                        int idx1 = (      j  * kernel_size      + i) * bottomchannels;
                        int idx2 = ((y2 + j) * bottomwidth + x2 + i) * bottomchannels;

                        for (int ch = 0; ch < bottomchannels; ch++)
                            sum += PrecisionUtils::f16tof32(patch_data[idx1 + ch]) * PrecisionUtils::f16tof32(bottom1[idx2 + ch]);
                    }
                }
                top[top_channel * topheight * topwidth + blockIdx_y * topwidth + blockIdx_x]
                    = PrecisionUtils::f32tof16(sum / (float)sumelems);
            }
        }
    }
}

static void refCorrelate(const Blob::Ptr in0,
                         const Blob::Ptr in1,
                         Blob::Ptr out,
                         int kernel_size, int max_displacement, int pad_size,
                         int stride1, int stride2) {
    // Correlation type = MULTIPLY
    ASSERT_NE(in0, nullptr);
    ASSERT_NE(in1, nullptr);
    ASSERT_NE(out, nullptr);

    const ie_fp16 *in0_data = in0->buffer();
    const ie_fp16 *in1_data = in1->buffer();
    ie_fp16 *out_data = out->buffer();
    ASSERT_NE(in0_data, nullptr);
    ASSERT_NE(in1_data, nullptr);
    ASSERT_NE(out_data, nullptr);

    int32_t IW0 = 1;
    int32_t IH0 = 1;
    int32_t IC0 = 1;
    get_dims(in0, IW0, IH0, IC0);
    int32_t IW1 = 1;
    int32_t IH1 = 1;
    int32_t IC1 = 1;
    get_dims(in1, IW1, IH1, IC1);
    ASSERT_EQ(IW0, IW1);
    ASSERT_EQ(IH0, IH1);
    ASSERT_EQ(IC0, IC1);

    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    get_dims(out, OW, OH, OC);

    const int bottomchannels = IC0;

    const int paddedbottomwidth  = IW0 + 2 * pad_size;
    const int paddedbottomheight = IH0 + 2 * pad_size;

    const int kernel_radius = kernel_size / 2; //size of unreachable border region (on each side)
    const int border_size = max_displacement + kernel_radius; //size of unreachable border region (on each side)

    const int top_width  = (int)ceilf((float)(paddedbottomwidth  - border_size * 2) / (float)stride1);
    const int top_height = (int)ceilf((float)(paddedbottomheight - border_size * 2) / (float)stride1);

    ASSERT_TRUE(top_width >= 1 && top_height >= 1)
        << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";

    // Given a center position in image 1,
    // how many displaced positions in -x / +x direction do we consider in image 2 (neighborhoodGridWidth):
    const int neighborhood_grid_radius = max_displacement / stride2;
    const int neighborhood_grid_width = 2 * neighborhood_grid_radius + 1;

    const int top_channels = neighborhood_grid_width * neighborhood_grid_width;

    ASSERT_TRUE(OC == top_channels && OH == top_height && OW == top_width)
        << "input and output blobs have incompatible shapes";

    auto rbot1 = std::vector<ie_fp16>(paddedbottomheight * paddedbottomwidth * bottomchannels);
    auto rbot2 = std::vector<ie_fp16>(paddedbottomheight * paddedbottomwidth * bottomchannels);

    const int bnum = 1;
    const int topcount = top_width * top_height * top_channels;

    const int pwidthheight = (IW0 + 2 * pad_size) * (IH0 + 2 * pad_size);

    rearrange(in0_data, rbot1.data(), bnum, IC0, IW0, IH0, IW0 * IH0, pad_size, pwidthheight);
    rearrange(in1_data, rbot2.data(), bnum, IC0, IW0, IH0, IW0 * IH0, pad_size, pwidthheight);

    const int height = IH0 + 2 * pad_size;
    const int width  = IW0  + 2 * pad_size;
    correlate(topcount, bnum, top_width, top_height, top_channels, topcount,
              max_displacement, neighborhood_grid_radius, neighborhood_grid_width,
              kernel_radius, kernel_size, stride1, stride2, width, height, IC0,
              rbot1.data(), rbot2.data(), out_data);
}

static float transform_forward_cpu(const ie_fp16* pic, const float px, const float py, int W, int H) {
    float res = 0.0f;
    float x = (px + 1) / 2 * H;
    float y = (py + 1) / 2 * W;
    int m, n, k, l;
    float w;
    k = (floorf(x));
    l = (floorf(y));
    m = floorf(x);
    n = floorf(y);
    w = 0;

    if (k >= 0 && k < H && l >= 0 && l < W) {
        w = fmaxf(0.0f, 1 - fabsf(x - m)) * fmaxf(0.0f, 1 - fabsf(y - n));
        res += w * PrecisionUtils::f16tof32(pic[k * W + l]);
    }

    k = (floorf(x) + 1);
    l = (floorf(y));
    m = floorf(x) + 1;
    n = floorf(y);

    w = 0;
    if (k >= 0 && k < H && l >= 0 && l < W) {
        w = fmaxf(0.0f, 1 - fabsf(x - m)) * fmaxf(0.0f, 1 - fabsf(y - n));
        res += w * PrecisionUtils::f16tof32(pic[k * W + l]);
    }
    k = (floorf(x));
    l = (floorf(y) + 1);
    m = floorf(x);
    n = floorf(y) + 1;
    w = 0;
    if (k >= 0 && k < H && l >= 0 && l < W) {
        w = fmaxf(0.0f, 1 - fabsf(x - m)) * fmaxf(0.0f, 1 - fabsf(y - n));
        res += w * PrecisionUtils::f16tof32(pic[k * W + l]);
    }
    k = (floorf(x) + 1);
    l = (floorf(y) + 1);
    m = floorf(x) + 1;
    n = floorf(y) + 1;
    w = 0;

    if (k >= 0 && k < H && l >= 0 && l < W) {
        w = fmaxf(0.0f, 1 - fabsf(x - m)) * fmaxf(0.0f, 1 - fabsf(y - n));
        res += w * PrecisionUtils::f16tof32(pic[k * W + l]);
    }

    return PrecisionUtils::f32tof16(res);
}

static void matrixMult(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
        const int m, const int n, const int k, const int transposeB) {
    if (transposeB) {
        for (int rowA = 0; rowA < m; rowA++) {
            for (int rowB = 0; rowB < n; rowB++) {
                float sum = 0;
                for (int colA = 0; colA < k; colA++) {
                    sum += A[rowA * k + colA] * B[rowB * k + colA];
                }
                C[rowA * n + rowB] = sum;
            }
        }
    } else {
        for (int rowA = 0; rowA < m; rowA++) {
            for (int colB = 0; colB < n; colB++) {
                float sum = 0;
                for (int colA = 0; colA < k; colA++) {
                    sum += A[rowA * k + colA] * B[colA * n + colB];
                }
                C[rowA * n + colB] = sum;
            }
        }
    }
}

static void refSpatialTransform(const Blob::Ptr& src, const Blob::Ptr& theta, Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(theta, nullptr);
    ASSERT_NE(dst, nullptr);

    const ie_fp16 *src_data = src->buffer();
    const ie_fp16 *theta_data = theta->buffer();
    ie_fp16 *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(theta_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    ASSERT_EQ(theta->size(), 6);

    int C = src->getTensorDesc().getDims()[1];
    int H = src->getTensorDesc().getDims()[2];
    int W = src->getTensorDesc().getDims()[3];

    auto input_grid_data = std::vector<float>(2*H*W);
    auto output_grid_data = std::vector<float>(3*H*W);
    auto theta_float = std::vector<float>(6);
    for (size_t i = 0; i < 6; i++) {
        theta_float[i] = PrecisionUtils::f16tof32(theta_data[i]);
    }

    for (int i = 0; i < H * W; ++i) {
        output_grid_data[3 * i] = ((i / W) * 1.0f / H * 2.0f - 1.0f);
        output_grid_data[3 * i + 1] = ((i % W) * 1.0f / W * 2.0f - 1.0f);
        output_grid_data[3 * i + 2] = 1.0f;
    }
    // Actually execute
    int M_size = H * W;
    int N_size = 2;
    int K_size = 3;
    matrixMult(output_grid_data, theta_float, input_grid_data, M_size, N_size, K_size, 1);
    for (int j = 0; j < C; ++j) {
        for (int s = 0; s < H; ++s) {
            for (int t = 0; t < W; ++t) {
                int row_idx = W * s + t;
                float px = input_grid_data[row_idx * 2 + 0];
                float py = input_grid_data[row_idx * 2 + 1];

                size_t dst_offset = (j * H + s) * W + t;
                size_t src_offset = (j * H + 0) * W + 0;
                dst_data[dst_offset] = transform_forward_cpu(src_data + src_offset, px, py, W, H);
            }
        }
    }
}

static std::vector<std::string> s_CustomConfig = {
#ifdef VPU_HAS_CUSTOM_KERNELS
    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

PRETTY_PARAM(Group, int)
PRETTY_PARAM(Levels, int)
PRETTY_PARAM(SwitchOut, int)
PRETTY_PARAM(Dilations, int)
PRETTY_PARAM(Kernel, param_size)
PRETTY_PARAM(Strides, int)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Group, std::string>> myriadLayersTestsShuffleChannel_smoke;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Levels, IRVersion, std::string>> myriadLayersTestsFakeQuantize_smoke;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Levels, SwitchOut, std::string>> myriadLayersTestsQuantizeBinarize_smoke;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Dilations, Group, Kernel, Strides, std::string>> myriadLayersTestsBinaryConvolution_smoke;
typedef myriadLayerTestBaseWithParam<std::tuple<std::vector<size_t>, std::string>> myriadLayersTestsExperimentalDetectronPriorGridGenerator_smoke;
typedef myriadLayerTestBaseWithParam<std::tuple<Dims, std::array<float, 6>, std::string>> myriadLayersTestsSpatialTransform_smoke;

struct CorrelateParams {
    tensor_test_params dims;
    int kernel_size;
    int pad_size;
    int max_displacement;
    int stride1;
    int stride2;
};

typedef myriadLayerTestBaseWithParam<std::tuple<CorrelateParams, std::string>> myriadLayersTestsCorrelate_smoke;

TEST_P(myriadLayersTestsShuffleChannel_smoke, ShuffleChannel) {
    tensor_test_params dims = std::get<0>(GetParam());
    int group                = std::get<1>(GetParam());
    std::string customConfig = std::get<2>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["group"] = std::to_string(group);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ShuffleChannel").params(params)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refShuffleChannel(_inputMap.begin()->second, _refBlob, group, false));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0);
}

static std::vector<Dims> s_ShuffleChannelTensors = {
    {{1,  48, 28, 28}},
    {{1,  96, 14, 14}},
    {{1, 192,  7,  7}},
};

static std::vector<Group> s_ShuffleChannelGroup = {
    2
};

TEST_P(myriadLayersTestsFakeQuantize_smoke, FakeQuantize) {
    tensor_test_params dims  = std::get<0>(GetParam());
    int levels               = std::get<1>(GetParam());
    _irVersion               = std::get<2>(GetParam());
    std::string customConfig = std::get<3>(GetParam());

    if (!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    srand(42);

    const auto inputFqSize = rand() % 2 ? 1 : dims.c;
    const auto outputFqSize = rand() % 2 ? 1 : dims.c;

    const auto inputDims = IN_OUT_desc{dims.asVector(),
        {1, inputFqSize, 1, 1},
        {1, inputFqSize, 1, 1},
        {1, outputFqSize, 1, 1},
        {1, outputFqSize, 1, 1}
    };

    SetInputTensors(inputDims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["levels"] = std::to_string(levels);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(
        LayerInitParams("FakeQuantize").params(params),
        NetworkInitParams()
             .layoutPreference(vpu::LayoutPreference::ChannelMajor)
             .lockLayout(true)));

    auto inputBlobs = std::vector<Blob::Ptr>{};
    inputBlobs.reserve(5);
    for (const auto& inputBlob : _inputMap) {
        inputBlobs.push_back(inputBlob.second);
    }

    const auto generateQuantBounds = [](const Blob::Ptr& lowBlob, const Blob::Ptr& highBlob) {
        IE_ASSERT(lowBlob->size() == highBlob->size());
        IE_ASSERT(lowBlob->getTensorDesc().getDims() == highBlob->getTensorDesc().getDims());

        const auto lowBound = lowBlob->buffer().as<ie_fp16 *>();
        const auto highBound = highBlob->buffer().as<ie_fp16 *>();
        for (std::size_t i = 0; i < lowBlob->size(); i++) {
        	const float val1 = rand() % 256;
        	const float val2 = 255.0f - fabs(val1);
        	lowBound[i] = PrecisionUtils::f32tof16(std::min(val1, val2));
        	highBound[i] = PrecisionUtils::f32tof16(std::max(val1, val2));
        }
    };

    generateQuantBounds(inputBlobs[1], inputBlobs[2]);
    generateQuantBounds(inputBlobs[3], inputBlobs[4]);

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refQuantize(inputBlobs[0],
                                        inputBlobs[1],
                                        inputBlobs[2],
                                        inputBlobs[3],
                                        inputBlobs[4],
                                        _refBlob,
                                        levels, true));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 1.f);
}

TEST_P(myriadLayersTestsQuantizeBinarize_smoke, Quantize_Binarization) {
    std::string model = R"V0G0N(
       <net name="Quantize_Binarization" version="2" batch="1">
           <layers>
            <layer id="0" name="data" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>@IB@</dim>
                        <dim>@IC@</dim>
                        <dim>@IH@</dim>
                        <dim>@IW@</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="input_low" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>@input_low_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="input_high" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>@input_high_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="3" name="output_low" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>@output_low_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="4" name="output_high" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>@output_high_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </output>
            </layer>
            <layer id="5" name="Quantize" precision="FP16" type="FakeQuantizeBin">
                <data levels="@levels@" input_low_size="@input_low_size@" input_high_size="@input_high_size@" output_low_size="@output_low_size@" output_high_size="@output_high_size@" switch_out="@switch_out@"/>
                <input>
                    <port id="0">
                        <dim>@IB@</dim>
                        <dim>@IC@</dim>
                        <dim>@IH@</dim>
                        <dim>@IW@</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>@input_low_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>@input_high_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                    <port id="3">
                        <dim>1</dim>
                        <dim>@output_low_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                    <port id="4">
                        <dim>1</dim>
                        <dim>@output_high_size@</dim>
                        <dim>1</dim>
                        <dim>1</dim>
                    </port>
                </input>
                <output>
                    <port id="0">
                        <dim>@OB@</dim>
                        <dim>@OC@</dim>
                        <dim>@OH@</dim>
                        <dim>@OW@</dim>
                    </port>
                </output>
            </layer>
           </layers>
           <edges>
               <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
               <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
               <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
               <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
               <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
           </edges>
       </net>
   )V0G0N";

    SetSeed(DEFAULT_SEED_VALUE + 6);

    tensor_test_params dims  = std::get<0>(GetParam());
    int levels               = std::get<1>(GetParam());
    int switch_out           = std::get<2>(GetParam());
    std::string customConfig = std::get<3>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    int IB = dims.n;
    int IC = dims.c;
    int IH = dims.h;
    int IW = dims.w;

    int OB = dims.n;
    int OC = dims.c;
    int OH = dims.h;
    int OW = dims.w;

    int input_low_size = (rand()%2>0) ? dims.c : 1;
    int input_high_size = (levels == 2) ? input_low_size : ((rand()%2>0) ? dims.c : 1);
    int output_low_size = (rand()%2>0) ? dims.c : 1;
    int output_high_size = (levels == 2) ? output_low_size : ((rand()%2>0) ? dims.c : 1);

    model.replace( model.find("@IB@"), sizeof("@IB@") -1, std::to_string(IB));
    model.replace( model.find("@IB@"), sizeof("@IB@") -1, std::to_string(IB));
    model.replace( model.find("@IC@"), sizeof("@IC@") -1, std::to_string(IC));
    model.replace( model.find("@IC@"), sizeof("@IC@") -1, std::to_string(IC));
    model.replace( model.find("@IH@"), sizeof("@IH@") -1, std::to_string(IH));
    model.replace( model.find("@IH@"), sizeof("@IH@") -1, std::to_string(IH));
    model.replace( model.find("@IW@"), sizeof("@IW@") -1, std::to_string(IW));
    model.replace( model.find("@IW@"), sizeof("@IW@") -1, std::to_string(IW));

    model.replace( model.find("@OB@"), sizeof("@OB@") -1, std::to_string(OB));
    model.replace( model.find("@OC@"), sizeof("@OC@") -1, std::to_string(OC));
    model.replace( model.find("@OH@"), sizeof("@OH@") -1, std::to_string(OH));
    model.replace( model.find("@OW@"), sizeof("@OW@") -1, std::to_string(OW));

    model.replace( model.find("@levels@"), sizeof("@levels@") -1, std::to_string(levels));
    model.replace( model.find("@switch_out@"), sizeof("@switch_out@") -1, std::to_string(switch_out));
    model.replace( model.find("@input_low_size@"), sizeof("@input_low_size@") -1, std::to_string(input_low_size));
    model.replace( model.find("@input_high_size@"), sizeof("@input_high_size@") -1, std::to_string(input_high_size));
    model.replace( model.find("@output_low_size@"), sizeof("@output_low_size@") -1, std::to_string(output_low_size));
    model.replace( model.find("@output_high_size@"), sizeof("@output_high_size@") -1, std::to_string(output_high_size));
    model.replace( model.find("@input_low_size@"), sizeof("@input_low_size@") -1, std::to_string(input_low_size));
    model.replace( model.find("@input_high_size@"), sizeof("@input_high_size@") -1, std::to_string(input_high_size));
    model.replace( model.find("@output_low_size@"), sizeof("@output_low_size@") -1, std::to_string(output_low_size));
    model.replace( model.find("@output_high_size@"), sizeof("@output_high_size@") -1, std::to_string(output_high_size));
    model.replace( model.find("@input_low_size@"), sizeof("@input_low_size@") -1, std::to_string(input_low_size));
    model.replace( model.find("@input_high_size@"), sizeof("@input_high_size@") -1, std::to_string(input_high_size));
    model.replace( model.find("@output_low_size@"), sizeof("@output_low_size@") -1, std::to_string(output_low_size));
    model.replace( model.find("@output_high_size@"), sizeof("@output_high_size@") -1, std::to_string(output_high_size));

    StatusCode st;

    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(model, InferenceEngine::Blob::CPtr());

    _inputsInfo  = network.getInputsInfo();
    _outputsInfo = network.getOutputsInfo();

    _inputsInfo["data"]->setPrecision(Precision::FP16);
    _inputsInfo["input_low"]->setPrecision(Precision::FP16);
    _inputsInfo["input_high"]->setPrecision(Precision::FP16);
    _inputsInfo["output_low"]->setPrecision(Precision::FP16);
    _inputsInfo["output_high"]->setPrecision(Precision::FP16);
    _outputsInfo["Quantize"]->setPrecision(Precision::FP16);

    _inputsInfo["data"]->setLayout(NCHW);
    _inputsInfo["input_low"]->setLayout(NCHW);
    _inputsInfo["input_high"]->setLayout(NCHW);
    _inputsInfo["output_low"]->setLayout(NCHW);
    _inputsInfo["output_high"]->setLayout(NCHW);
    _outputsInfo["Quantize"]->setLayout(NCHW);

    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network,
                                                    {{InferenceEngine::MYRIAD_CUSTOM_LAYERS, customConfig }}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr data;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("data", data, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(data);

    Blob::Ptr input_low;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("input_low", input_low, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(input_low);

    Blob::Ptr input_high;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("input_high", input_high, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    Blob::Ptr output_low;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("output_low", output_low, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    Blob::Ptr output_high;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("output_high", output_high, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    if(levels == 2){
        memcpy((uint8_t*)input_high->buffer(), (uint8_t*)input_low->buffer(), input_high->byteSize());
        for(int i = 0; i < (output_low->byteSize() / output_low->element_size()); ++i){
            *((ie_fp16*)output_low->buffer() + i) = switch_out ? PrecisionUtils::f32tof16(1.0f) : PrecisionUtils::f32tof16(-1.0f);
            *((ie_fp16*)output_high->buffer() + i) = switch_out ? PrecisionUtils::f32tof16(-1.0f) : PrecisionUtils::f32tof16(1.0f);
        }
    }
    else{
        GenRandomData(input_high);
        GenRandomData(output_low);
        GenRandomData(output_high);
    }

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

{
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    _inferRequest->GetPerformanceCounts(perfMap, nullptr);
    std::vector <std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
    std::sort(perfVec.begin(), perfVec.end(),
              [=](const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair1,
                  const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair2) -> bool {
                  return pair1.second.execution_index < pair2.second.execution_index;
              });

    unsigned currentIndex = 0;
    for (auto it = perfVec.begin(); it != perfVec.end(); ++it) {
        std::string layerName = it->first;
        InferenceEngine::InferenceEngineProfileInfo info = it->second;
        if (info.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
            printf("\x1B[32m[----------]\x1B[0m Myriad time = '%s' layer with '%s' type is %f ms.\n", layerName.c_str(), info.exec_type, info.realTime_uSec / 1000.f);
        }
    }
}

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(_inferRequest->GetBlob("Quantize", outputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    _refBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, outputBlob->getTensorDesc().getDims(), NCHW));
    _refBlob->allocate();

    ASSERT_NO_FATAL_FAILURE(ref_QuantizeBinarization(data,
                                                    input_low,
                                                    input_high,
                                                    output_low,
                                                    output_high,
                                                    _refBlob,
                                                    levels));

    CompareCommonAbsolute(outputBlob, _refBlob, 0.1);
}

static std::vector<Dims> s_QuantizeTensors = {
    {{1,  64, 56, 56}},
    {{1, 256, 28, 28}},
    {{1, 512,  7,  7}},
    {{1,  64, 56, 57}},
    {{1, 256, 28, 31}},
    {{1, 512,  8,  9}},
    {{1,  64, 56, 56}},
    {{1, 256, 56, 56}},
    {{1, 128, 56, 56}},
    {{1, 128, 28, 28}},
    {{1, 512, 28, 28}},
    {{1, 256, 28, 28}},
    {{1, 256, 14, 14}},
    {{1, 1024,14, 14}},
    {{1, 512, 14, 14}},
    {{1, 512,  7,  7}},
    {{1, 2048, 7,  7}},
    {{1, 512,  7,  7}}
};

static std::vector<Levels> s_QuantizeLevels = {
    2,
    256
};

static std::vector<SwitchOut> s_QuantizeSwitchOut = {
    0,
    1
};

TEST_P(myriadLayersTestsBinaryConvolution_smoke, BinaryConvolution) {
    tensor_test_params dims  = std::get<0>(GetParam());
    int dilations            = std::get<1>(GetParam());
    int group                = std::get<2>(GetParam());
    param_size kernel        = std::get<3>(GetParam());
    int strides              = std::get<4>(GetParam());
    std::string customConfig = std::get<5>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    SetInputTensor(dims);
    auto dimsOutput = dims;
    dimsOutput.h = (dims.h) / strides;
    dimsOutput.w = (dims.w) / strides;
    SetOutputTensor(dimsOutput);
    size_t numWeights = kernel.x * kernel.y * dims.c * dims.c;
    size_t numBiases = 0;
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(numWeights));

    std::map<std::string, std::string> params;
    params["mode"] = "xnor-popcount";
    params["pad_value"] = "-1.0";
    params["pads_begin"] = std::to_string(kernel.x/2) + "," + std::to_string(kernel.y/2);
    params["pads_end"] = std::to_string(kernel.x/2) + "," + std::to_string(kernel.y/2);
    params["input"] = std::to_string(dims.c);
    params["output"] = std::to_string(dims.c);
    params["dilations"] = std::to_string(dilations) + "," + std::to_string(dilations);
    params["group"] = std::to_string(group);
    params["kernel"] = std::to_string(kernel.x) + "," + std::to_string(kernel.y);
    params["strides"] = std::to_string(strides) + "," + std::to_string(strides);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("BinaryConvolution")
                                        .params(params)
                                        .weights(numWeights)
                                        .biases(numBiases),
                                        {},
                                        weights_ptr));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refBinaryConvolution(_inputMap.begin()->second, weights_ptr, _refBlob,
                                                 dilations, group, kernel, strides,
                                                 false));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0);
}

static std::vector<Dims> s_BinaryConvolutionTensors = {
    {{1, 64, 112, 112}},
    {{1, 128, 56, 56}},
    {{1, 256, 28, 28}},
    {{1, 256, 14, 14}},
    {{1, 16, 16, 16}},
    {{1,  2,  2,  2}},
};

static std::vector<Dilations> s_BinaryConvolutionDilations = {
    1, 2
};
static std::vector<Group> s_BinaryConvolutionGroup = {
    1, 2
};
static std::vector<Kernel> s_BinaryConvolutionKernel = {
    {{1, 1}},
    {{1, 3}},
    {{3, 3}}
};
static std::vector<Strides> s_BinaryConvolutionStrides = {
    1, 2
};

TEST_P(myriadLayersTestsExperimentalDetectronPriorGridGenerator_smoke,
       ExperimentalDetectronPriorGridGenerator) {

    // Setup parameters and configuration.
    std::vector<size_t> image_dims = std::get<0>(GetParam());
    std::string customConfig = std::get<1>(GetParam());
    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    IN_OUT_desc inputTensors = {{1, 1, 3, 4}, image_dims, {1, 3, 480, 480}};
    IN_OUT_desc outputTensors = {{1, 1,
         inputTensors[0][2] *
         inputTensors[1][2] *
         inputTensors[1][3],
         inputTensors[0][3]}};
    SetInputTensors(inputTensors);
    SetOutputTensors(outputTensors);

    // Calculate strides. The stride dimensions are calculated by the equation
    // (image feature map dimension) / (input feature map dimension).
    float stride_h = static_cast<float>(inputTensors[2][2]) /
                     inputTensors[1][2];
    float stride_w = static_cast<float>(inputTensors[2][3]) /
                     inputTensors[1][3];

    std::map<std::string, std::string> params = {
        {"stride_h", std::to_string(stride_h)},
        {"stride_w", std::to_string(stride_w)}
    };
    // Run inference on OpenCL kernel.
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(
                LayerInitParams("ExperimentalDetectronPriorGridGenerator").params(params)));
    ASSERT_TRUE(Infer());

    // Setup of reference input and reference output blobs.
    std::vector<Blob::Ptr> reference_input_blobs(inputTensors.size());
    std::vector<Blob::Ptr> reference_output_blobs(outputTensors.size());
    int k = 0;
    for (auto& p : _inputMap) {
        reference_input_blobs[k++] = p.second;
    }
    reference_output_blobs[0] = _refBlob;

    // Run inference on reference implementation.
    refExperimentalDetectronPriorGridGenerator(
            reference_input_blobs, reference_output_blobs,
            inputTensors[1][2], inputTensors[1][3], stride_h, stride_w);

    CompareCommonAbsolute(_outputMap.begin()->second, reference_output_blobs[0], 0.01f);
}

static std::vector<std::vector<size_t>>
s_ExperimentalDetectronPriorGridGeneratorImageDims = {
    {1, 128, 240, 240},
    {1, 128, 120, 120},
    {1, 128, 60, 60},
    {1, 128, 30, 30}
};

TEST_P(myriadLayersTestsCorrelate_smoke, Correlate) {
    const auto test = std::get<0>(GetParam());
    const auto dims = test.dims;
    const int kernel_size = test.kernel_size;
    const int pad_size = test.pad_size;
    const int max_displacement = test.max_displacement;
    const int stride1 = test.stride1;
    const int stride2 = test.stride2;
    const std::string customConfig = std::get<1>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    const int paddedbottomwidth  = dims.w + 2 * pad_size;
    const int paddedbottomheight = dims.h + 2 * pad_size;

    const int kernel_radius = kernel_size / 2; //size of unreachable border region (on each side)
    const int border_size = max_displacement + kernel_radius; //size of unreachable border region (on each side)

    const int neighborhood_grid_radius = max_displacement / stride2;
    const int neighborhood_grid_width = 2 * neighborhood_grid_radius + 1;

    const int top_width  = (int)ceilf((float) (paddedbottomwidth - border_size * 2) / (float) stride1);
    const int top_height = (int)ceilf((float)(paddedbottomheight - border_size * 2) / (float)stride1);
    const int top_channels = (test.max_displacement + 1) * (test.max_displacement + 1);// neighborhood_grid_width * neighborhood_grid_width;

    const auto inputTensors = IN_OUT_desc{dims.asVector(), dims.asVector()};
    const auto outputTensors = IN_OUT_desc{{1, (uint32_t)top_channels, (uint32_t)top_height, (uint32_t)top_width}};

    SetInputTensors(inputTensors);
    SetOutputTensors(outputTensors);

    std::map<std::string, std::string> params = {
        {"top_width", std::to_string(top_width)},
        {"top_height", std::to_string(top_height)},
        {"width", std::to_string(dims.w)},
        {"height", std::to_string(dims.h)},
        {"channels", std::to_string(dims.c)},
        {"displacement", std::to_string(max_displacement)},
        {"pad", std::to_string(pad_size)},
        {"neighborhood_grid_radius", std::to_string(neighborhood_grid_radius)},
        {"neighborhood_grid_width", std::to_string(neighborhood_grid_width)},
        {"kernel_size", std::to_string(kernel_size)},
        {"stride", std::to_string(stride1) + "," + std::to_string(stride2)},
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(
                LayerInitParams("Correlate").params(params),
                NetworkInitParams()
                .layoutPreference(vpu::LayoutPreference::ChannelMajor)
                .lockLayout(true)));

    std::vector<Blob::Ptr> input_blobs{};
    input_blobs.reserve(_inputMap.size());
    for (auto& input : _inputMap) {
        // generate input data
        for (int i = 0; i < dims.c * dims.h * dims. w; i++) {
            const float corr_min = -1.744443f;
            const float corr_max = 11.167725f;
            float val = (corr_min + (float) rand() / ((float) RAND_MAX / (corr_max - corr_min + 1.f) + 1.f));

            auto buf = input.second->buffer().as<ie_fp16*>();
            buf[i] = PrecisionUtils::f32tof16(val);
        }

        input_blobs.push_back(input.second);
    }
    const int output_size = top_width * top_height * top_channels;
    for (int i = 0; i < output_size; i++) {
        _outputMap.begin()->second->buffer().as<ie_fp16*>()[i] = 0;
        _refBlob->buffer().as<ie_fp16*>()[i] = 0;
    }

    ASSERT_TRUE(Infer());

    refCorrelate(input_blobs[0], input_blobs[1], _refBlob, kernel_size, max_displacement, pad_size, stride1, stride2);

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0.1f);
}

static const std::vector<CorrelateParams> s_CorrelateParams = {
    { {1, 64, 48, 64}, 1, 8, 8, 1, 2 },
    { {1, 127, 12, 64}, 3, 8, 8, 1, 2 },
    { {1, 256, 48, 64}, 1, 20, 20, 1, 2 }
};

TEST_P(myriadLayersTestsSpatialTransform_smoke, SpatialTransform) {
    const tensor_test_params dims = std::get<0>(GetParam());
    const std::array<float, 6> theta = std::get<1>(GetParam());
    const std::string customConfig = std::get<2>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    SetInputTensors({dims.asVector(), {1, 1, 2, 3}});
    SetOutputTensor(dims);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(
                LayerInitParams("SpatialTransform"),
                NetworkInitParams()
                    .layoutPreference(vpu::LayoutPreference::ChannelMajor)
                    .lockLayout(true)));

    auto theta_half = std::next(_inputMap.begin())->second;
    for (int i = 0; i < 6; i++) {
        theta_half->buffer().as<ie_fp16*>()[i] = PrecisionUtils::f32tof16(theta[i]);
    }

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refSpatialTransform(_inputMap.begin()->second,
                            std::next(_inputMap.begin())->second,
                            _refBlob));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0.001f);
}

static const std::vector<Dims> s_SpatialTransformInputs = {
	{{ 1, 3,  24,  94 }},
	{{ 1, 3,  96, 188 }},
	{{ 1, 3,  97, 189 }},
	{{ 1, 3,  98, 190 }},
	{{ 1, 3, 384, 512 }},
	{{ 1, 3,  24, 640 }},
};

static const std::vector<std::array<float, 6>> s_SpatialTransformTheta = {
	{1.2f, 0.2f, -0.2f, 0.2f, 1.2f, -0.2f},
	{1.f, 0.f, 0.f, 0.0f, 1.f, 0.f}
};
