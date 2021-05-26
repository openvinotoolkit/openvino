// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <legacy/ie_layers.h>
#include <precision_utils.h>
#include "conv_ref.hpp"
#include "common_layers_params.hpp"

using namespace InferenceEngine;


void Convolution_parseParams(InferenceEngine::CNNLayer* layer) {
    auto convLayer = dynamic_cast<InferenceEngine::ConvolutionLayer*>(layer);
    if (!convLayer) {
        IE_THROW() << "Layer is not instance of ConvolutionLayer class";
    }
    convLayer->_out_depth = convLayer->GetParamAsUInt("output");

    convLayer->_kernel.clear();
    convLayer->_stride.clear();
    convLayer->_padding.clear();
    convLayer->_pads_end.clear();
    convLayer->_dilation.clear();

    std::vector<unsigned int> kernels = convLayer->GetParamAsUInts("kernel", {});
    if (kernels.empty()) {
        // IR_v == 2
        convLayer->_kernel.insert(InferenceEngine::X_AXIS, convLayer->GetParamAsUInt("kernel-x"));
        convLayer->_kernel.insert(InferenceEngine::Y_AXIS, convLayer->GetParamAsUInt("kernel-y"));

        convLayer->_stride.insert(InferenceEngine::X_AXIS, convLayer->GetParamAsUInt("stride-x", 1u));
        convLayer->_stride.insert(InferenceEngine::Y_AXIS, convLayer->GetParamAsUInt("stride-y", 1u));
        // TODO: maybe just throw exception, why do we change IR?
        if (0 == convLayer->_stride[InferenceEngine::X_AXIS]) {
            convLayer->_stride[InferenceEngine::X_AXIS] = 1u;
            printf("Warning! in layer %s: Stride x is 0, setting to 1 ", convLayer->name.c_str());
        }
        if (0 == convLayer->_stride[InferenceEngine::Y_AXIS]) {
            convLayer->_stride[InferenceEngine::Y_AXIS] = 1u;
            printf("Warning! in layer %s: Stride y is 0, setting to 1", convLayer->name.c_str());
        }

        convLayer->_padding.insert(InferenceEngine::X_AXIS, convLayer->GetParamAsUInt("pad-x", 0u));
        convLayer->_padding.insert(InferenceEngine::Y_AXIS, convLayer->GetParamAsUInt("pad-y", 0u));

        convLayer->_pads_end.insert(InferenceEngine::X_AXIS, convLayer->GetParamAsUInt("pad-r", convLayer->_padding[InferenceEngine::X_AXIS]));
        convLayer->_pads_end.insert(InferenceEngine::Y_AXIS, convLayer->GetParamAsUInt("pad-b", convLayer->_padding[InferenceEngine::Y_AXIS]));

        convLayer->_dilation.insert(InferenceEngine::X_AXIS, convLayer->GetParamAsUInt("dilation-x", 1u));
        convLayer->_dilation.insert(InferenceEngine::Y_AXIS, convLayer->GetParamAsUInt("dilation-y", 1u));
    } else {
        // IR_v > 2
        for (size_t i = 1; i <= kernels.size(); i++) {
            convLayer->_kernel.insert(i - 1, kernels[kernels.size() - i]);
        }

        std::vector<unsigned int> default_0 = std::vector<unsigned int> (convLayer->_kernel.size(), 0u);
        std::vector<unsigned int> default_1 = std::vector<unsigned int> (convLayer->_kernel.size(), 1u);

        std::vector<unsigned int> strides = convLayer->GetParamAsUInts("strides", default_1);
        for (size_t i = 1; i <= strides.size(); i++) {
            if (strides[strides.size() - i] == 0) {
                IE_THROW() << "Stride could not be 0.\nIn layer " << convLayer->name;
            }
            convLayer->_stride.insert(i - 1, strides[strides.size() - i]);
        }

        std::vector<unsigned int> pads_begin = convLayer->GetParamAsUInts("pads_begin", default_0);
        for (size_t i = 1; i <= pads_begin.size(); i++) {
            convLayer->_padding.insert(i - 1, pads_begin[pads_begin.size() - i]);
        }

        std::vector<unsigned int> pads_end = convLayer->GetParamAsUInts("pads_end", pads_begin);
        for (size_t i = 1; i <= pads_end.size(); i++) {
            convLayer->_pads_end.insert(i - 1, pads_end[pads_end.size() - i]);
        }

        std::vector<unsigned int> dilations = convLayer->GetParamAsUInts("dilations", default_1);
        for (size_t i = 1; i <= dilations.size(); i++) {
            convLayer->_dilation.insert(i - 1, dilations[dilations.size() - i]);
        }
    }

    convLayer->_auto_pad = convLayer->GetParamAsString("auto_pad", "");
    convLayer->_group = convLayer->GetParamAsUInt("group", 1u);
}

template<typename wei_data_t, typename bias_data_t>
void ref_conv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                     Blob& dst,
                     const wei_data_t* weights_data,
                     size_t weights_size,
                     const bias_data_t* bias_data,
                     size_t bias_size,
                     const CommonTestUtils::conv_common_params& prm) {
    if (srcs[0]->getTensorDesc().getLayout() != Layout::NCHW &&
            srcs[0]->getTensorDesc().getLayout() != Layout::NCDHW)
        IE_THROW() << "Reference FP32 convolution supports NCHW and NCDHW layouts only";
    size_t KW = prm.kernel[X_AXIS];
    size_t KH = prm.kernel[Y_AXIS];
    size_t KD = prm.kernel.size() > Z_AXIS ? prm.kernel[Z_AXIS] : 1lu;

    size_t SW = prm.stride[X_AXIS];
    size_t SH = prm.stride[Y_AXIS];
    size_t SD = prm.stride.size() > Z_AXIS ? prm.stride[Z_AXIS] : 0lu;

    size_t DW = prm.dilation[X_AXIS];
    size_t DH = prm.dilation[Y_AXIS];
    size_t DD = prm.dilation.size() > Z_AXIS ? prm.dilation[Z_AXIS] : 0lu;

    size_t PW = prm.pads_begin[X_AXIS];
    size_t PH = prm.pads_begin[Y_AXIS];
    size_t PD = prm.pads_begin.size() > Z_AXIS ? prm.pads_begin[Z_AXIS] : 0lu;

    size_t GC = prm.group;

    auto src_dims = srcs[0]->getTensorDesc().getDims();
    size_t IC = src_dims[1];
    size_t ID = (src_dims.size() == 5lu) ? src_dims[2] : 1lu;
    size_t IH = src_dims.at(src_dims.size() - 2);
    size_t IW = src_dims.back();

    auto dst_dims = dst.getTensorDesc().getDims();
    size_t OW = dst_dims.back();
    size_t OH = dst_dims.at(dst_dims.size() - 2);
    size_t OD = (dst_dims.size() == 5lu) ? dst_dims[2] : 1lu;
    size_t OC = prm.out_c;

    const auto src_buffer = srcs[0]->cbuffer();
    auto* dst_data = dst.buffer().as<float*>();
    Precision src_precision = srcs[0]->getTensorDesc().getPrecision();

    IE_ASSERT(KW * KH * KD * OC * IC / GC == weights_size);
    IE_ASSERT(OC == bias_size);

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t od = 0; od < OD; od++) {
                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        size_t oidx = g * OC / GC * OD * OH * OW
                                      + oc * OD * OH * OW
                                      + od * OH * OW
                                      + oh * OW
                                      + ow;
                        if (bias_data)
                            dst_data[oidx] = bias_data[g * OC / GC + oc];

                        for (size_t ic = 0; ic < IC / GC; ic++) {
                            for (size_t kd = 0; kd < KD; kd++) {
                                for (size_t kh = 0; kh < KH; kh++) {
                                    for (size_t kw = 0; kw < KW; kw++) {
                                        int32_t iw = ow * SW - PW + kw * DW;
                                        int32_t ih = oh * SH - PH + kh * DH;
                                        int32_t id = od * SD - PD + kd * DD;
                                        if (iw < 0 || iw >= (int32_t) IW ||
                                            ih < 0 || ih >= (int32_t) IH ||
                                            id < 0 || id >= (int32_t) ID)
                                            continue;
                                        size_t iidx = g * IC / GC * ID * IH * IW
                                                      + ic * ID * IH * IW
                                                      + id * IH * IW
                                                      + ih * IW
                                                      + iw;
                                        size_t widx = g * OC / GC * IC / GC * KD * KH * KW
                                                      + oc * IC / GC * KD * KH * KW
                                                      + ic * KD * KH * KW
                                                      + kd * KH * KW
                                                      + kh * KW
                                                      + kw;

                                        if (src_precision == Precision::U8) {
                                            dst_data[oidx] += (src_buffer.as<const uint8_t*>())[iidx] * weights_data[widx];
                                        } else if (src_precision == Precision::I8) {
                                            dst_data[oidx] += (src_buffer.as<const int8_t*>())[iidx] * weights_data[widx];
                                        } else {
                                            dst_data[oidx] += (src_buffer.as<const float*>())[iidx] * weights_data[widx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template void ref_conv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob& dst, const float* weights_data,
                              size_t, const float* bias_data, size_t, const CommonTestUtils::conv_common_params& prm);
template void ref_conv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob& dst, const int8_t* weights_data,
                              size_t, const int32_t* bias_data, size_t, const CommonTestUtils::conv_common_params& prm);

template<>
void ref_conv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                     Blob& dst,
                     const ie_fp16* weights_data,
                     size_t /*weights_size*/,
                     const ie_fp16* bias_data,
                     size_t /*bias_size*/,
                     const CommonTestUtils::conv_common_params& prm) {
    const auto* src_data = srcs[0]->cbuffer().as<const ie_fp16*>();
    auto* dst_data = dst.buffer().as<ie_fp16*>();
    IE_ASSERT(src_data != nullptr);
    IE_ASSERT(dst_data != nullptr);

    size_t KH = prm.kernel[Y_AXIS];
    size_t KW = prm.kernel[X_AXIS];

    size_t SH = prm.stride[Y_AXIS];
    size_t SW = prm.stride[X_AXIS];

    size_t DH = prm.dilation[Y_AXIS];
    size_t DW = prm.dilation[X_AXIS];

    size_t PH = prm.pads_begin[Y_AXIS];
    size_t PW = prm.pads_begin[X_AXIS];

    size_t GC = prm.group;

    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    int32_t OW = 0;
    int32_t OH = 0;
    int32_t OC = 0;
    int32_t ON = 0;
    CommonTestUtils::get_common_dims(*srcs[0], IW, IH, IC, I_N);
    CommonTestUtils::get_common_dims(dst, OW, OH, OC, ON);
    IE_ASSERT(I_N == ON);
    size_t src_channels = IC / GC;
    size_t dst_channels = OC / GC;
    for (size_t n = 0; n < ON; ++n) {
        size_t oShift = n * OC * OH * OW;
        size_t iShift = n * IC * IH * IW;
        for (size_t g = 0; g < GC; ++g) {
            for (size_t oc = 0; oc < dst_channels; ++oc) {
                size_t dst_channel = (g * dst_channels + oc);
                for (size_t oh = 0; oh < OH; oh++) {
                    for (size_t ow = 0; ow < OW; ow++) {
                        size_t oidx = dst_channel + ow * OC + oh * OC * OW + oShift;
                        IE_ASSERT(oidx < dst.size());
                        float val = 0.0f;
                        if (bias_data)
                            val = PrecisionUtils::f16tof32(bias_data[dst_channel]);

                        for (size_t ic = 0; ic < src_channels; ++ic) {
                            size_t src_channel = (g * src_channels + ic);

                            for (size_t ky = 0; ky < KH; ++ky) {
                                for (size_t kx = 0; kx < KW; ++kx) {

                                    int32_t iw = ow * SW - PW + kx * DW;
                                    int32_t ih = oh * SH - PH + ky * DH;

                                    if (iw < 0 || iw >= (int32_t) IW || ih < 0 || ih >= (int32_t) IH) {
                                        continue;
                                    }

                                    size_t iidx = src_channel + iw * IC + ih * IC * IW + iShift;
                                    IE_ASSERT(iidx < srcs[0]->size());

                                    size_t widx = (ky * KW + kx) + ic * KH * KW +
                                                  dst_channel * src_channels * KW * KH;

                                    IE_ASSERT(widx < KH * KW * (IC / GC) * OC);

                                    val += PrecisionUtils::f16tof32(src_data[iidx]) *
                                           PrecisionUtils::f16tof32(weights_data[widx]);
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
