// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <legacy/ie_layers.h>
#include <precision_utils.h>
#include <gtest/gtest.h>
#include "deconv_ref.hpp"
#include "common_layers_params.hpp"

using namespace InferenceEngine;

template<>
void ref_deconv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                       Blob &dst,
                       const float *weights_data,
                       size_t weights_size,
                       const float *bias_data,
                       size_t bias_size,
                       const CommonTestUtils::conv_common_params &prm) {
    if (srcs[0]->getTensorDesc().getLayout() != Layout::NCHW)
        IE_THROW() << "Reference FP32 convolution supports NCHW layout only";

    size_t KH = prm.kernel[Y_AXIS];
    size_t KW = prm.kernel[X_AXIS];

    size_t SH = prm.stride[Y_AXIS];
    size_t SW = prm.stride[X_AXIS];

    size_t PH = prm.pads_begin[Y_AXIS];
    size_t PW = prm.pads_begin[X_AXIS];

    auto src_dims = srcs[0]->getTensorDesc().getDims();
    size_t IW = src_dims.back();
    size_t IH = src_dims.at(src_dims.size() - 2);
    size_t IC = src_dims.at(1);
    size_t MB = src_dims.at(0);

    size_t OC = prm.out_c;

    auto dst_dims = dst.getTensorDesc().getDims();
    size_t OW = dst_dims.back();
    size_t OH = dst_dims.at(dst_dims.size() - 2);

    const auto *src_data = srcs[0]->cbuffer().as<float *>();
    auto *dst_data = dst.buffer().as<float *>();;

    for (int mb = 0; mb < MB; ++mb) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    size_t didx = mb * OC * OH * OW
                                  + oc * OH * OW + oh * OW + ow;

                    dst_data[didx] = float(0);
                    if (bias_data) dst_data[didx] += bias_data[oc];

                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                if (ow + PW < kw || oh + PH < kh)
                                    continue;

                                size_t iw = ow - kw + PW;
                                size_t ih = oh - kh + PH;

                                if (iw % SW != 0 || ih % SH != 0)
                                    continue;

                                iw /= SW;
                                ih /= SH;

                                if (ih < IH && iw < IW) {
                                    size_t sidx = mb * IC * IH * IW
                                                  + ic * IH * IW + ih * IW
                                                  + iw;

                                    size_t widx = ic * OC * KH * KW
                                                  + oc * KH * KW + kh * KW
                                                  + kw;

                                    dst_data[didx] += src_data[sidx] * weights_data[widx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<>
void ref_deconv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                       Blob &dst,
                       const ie_fp16 *weights_data,
                       size_t /*weights_size*/,
                       const ie_fp16 *bias_data,
                       size_t /*bias_size*/,
                       const CommonTestUtils::conv_common_params &prm) {
    const auto *src_data = srcs[0]->cbuffer().as<ie_fp16 *>();
    auto *dst_data = dst.buffer().as<ie_fp16 *>();
    IE_ASSERT(src_data != nullptr);
    IE_ASSERT(dst_data != nullptr);

    size_t KH = prm.kernel[Y_AXIS];
    size_t KW = prm.kernel[X_AXIS];

    size_t SH = prm.stride[Y_AXIS];
    size_t SW = prm.stride[X_AXIS];

    size_t PH = prm.pads_begin[Y_AXIS];
    size_t PW = prm.pads_begin[X_AXIS];

    auto src_dims = srcs[0]->getTensorDesc().getDims();
    size_t IW = src_dims.back();
    size_t IH = src_dims.at(src_dims.size() - 2);
    size_t IC = src_dims.at(1);
    size_t IB = src_dims.at(0);

    auto dst_dims = dst.getTensorDesc().getDims();
    size_t OW = dst_dims.back();
    size_t OH = dst_dims.at(dst_dims.size() - 2);
    size_t OC = dst_dims.at(1);
    size_t OB = src_dims.at(0);

    size_t GC = prm.group;

    size_t src_channels = IC / GC;
    size_t dst_channels = OC / GC;

    size_t ib_size = srcs[0]->size() / IB;
    size_t ob_size = dst.size() / OB;

    for (size_t ob = 0; ob < OB; ++ob) {
        for (size_t g = 0; g < GC; ++g) {
            for (size_t oc = 0; oc < dst_channels; ++oc) {
                size_t dst_channel = (g * dst_channels + oc);
                for (size_t oy = 0; oy < OH; oy++) {
                    for (size_t ox = 0; ox < OW; ox++) {
                        size_t oidx = ob * ob_size + dst_channel + ox * OC + oy * OC * OW;
                        ASSERT_LT(oidx, dst.size());
                        float val = bias_data != nullptr ? PrecisionUtils::f16tof32(bias_data[dst_channel]) : 0;

                        for (size_t ic = 0; ic < src_channels; ++ic) {
                            size_t src_channel = (g * src_channels + ic);

                            for (size_t ky = 0; ky < KH; ++ky) {
                                for (size_t kx = 0; kx < KW; ++kx) {
                                    if (ox + PW < kx || oy + PH < ky)
                                        continue;

                                    int32_t ix = ox - kx + PW;
                                    int32_t iy = oy - ky + PH;

                                    if (ix % SW != 0 || iy % SH != 0)
                                        continue;

                                    ix /= SW;
                                    iy /= SH;

                                    if (iy < IH && ix < IW) {
                                        size_t iidx = ob * ib_size + src_channel + ix * IC + iy * IC * IW;

                                        ASSERT_LT(iidx, srcs[0]->size());

                                        size_t widx = ic * OC * KH * KW
                                                    + dst_channel * KH * KW
                                                    + ky * KW
                                                    + kx;

                                        ASSERT_LT(widx, KW * KH * (IC / GC) * OC);

                                        val += PrecisionUtils::f16tof32(src_data[iidx]) *
                                            PrecisionUtils::f16tof32(weights_data[widx]);
                                    }
                                }
                            }
                        }

                        dst_data[oidx] = PrecisionUtils::f32tof16(val);
                    }
                }
            }
        }
    }
}
