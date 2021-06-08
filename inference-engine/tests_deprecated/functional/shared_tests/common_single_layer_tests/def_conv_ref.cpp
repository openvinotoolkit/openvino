// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <legacy/ie_layers.h>
#include <precision_utils.h>
#include <math.h>
#include <ie_parallel.hpp>
#include "def_conv_ref.hpp"
#include "common_layers_params.hpp"

using namespace InferenceEngine;

void Convolution_parseParams(InferenceEngine::CNNLayer* layer);

void DeformableConvolution_parseParams(InferenceEngine::CNNLayer* layer) {
    auto deformable_conv_layer = dynamic_cast<InferenceEngine::DeformableConvolutionLayer*>(layer);
    if (!deformable_conv_layer) {
        IE_THROW() << "Layer is not instance of DeformableConvolutionLayer class";
    }
    deformable_conv_layer->_deformable_group = deformable_conv_layer->GetParamAsUInt("deformable_group", 1u);
    Convolution_parseParams(layer);
}

template<>
void ref_def_conv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                         Blob& dst,
                         const float* weights_data,
                         size_t weights_size,
                         const float* bias_data,
                         size_t bias_size,
                         const CommonTestUtils::def_conv_common_params& prm) {
    if (srcs[0]->getTensorDesc().getLayout() != Layout::NCHW &&
        srcs[0]->getTensorDesc().getLayout() != Layout::NCDHW)
        IE_THROW() << "Reference FP32 deformable convolution supports NCHW and NCDHW layouts only";
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
    size_t MB = src_dims[0];
    size_t IC = src_dims[1];
    size_t ID = (src_dims.size() == 5lu) ? src_dims[2] : 1lu;
    size_t IH = src_dims.at(src_dims.size() - 2);
    size_t IW = src_dims.back();

    auto dst_dims = dst.getTensorDesc().getDims();
    size_t OW = dst_dims.back();
    size_t OH = dst_dims.at(dst_dims.size() - 2);
    size_t OD = (dst_dims.size() == 5lu) ? dst_dims[2] : 1lu;
    size_t OC = prm.out_c;

    size_t DG = prm.deformable_group;

    const auto* src_data = srcs[0]->cbuffer().as<const float*>();
    const auto* trans_data = srcs[1]->cbuffer().as<const float*>();
    auto* dst_data = dst.buffer().as<float*>();

    IE_ASSERT(KW * KH * KD * OC * IC / GC == weights_size);
    IE_ASSERT(OC == bias_size);

    const int channel_per_deformable_group = IC / DG;

    parallel_for5d(MB, GC, OC / GC, OD, OH, [&](size_t mb, size_t g, size_t oc, size_t od, size_t oh) {
        for (size_t ow = 0; ow < OW; ow++) {
            size_t oidx = mb * OC * OD * OH * OW
                          + g * OC / GC * OD * OH * OW
                          + oc * OD * OH * OW
                          + od * OH * OW
                          + oh * OW
                          + ow;
            if (bias_data)
                dst_data[oidx] = bias_data[g * OC / GC + oc];

            for (size_t ic = 0; ic < IC / GC; ic++) {
                const int deformable_group_idx = ic / channel_per_deformable_group;
                const int trans_offset = mb * DG * 2 * KH * KW * OH * OW
                                         + deformable_group_idx * 2 * KH * KW * OH * OW;

                for (size_t kd = 0; kd < KD; kd++) {
                    for (size_t kh = 0; kh < KH; kh++) {
                        for (size_t kw = 0; kw < KW; kw++) {
                            int32_t iw = ow * SW - PW + kw * DW;
                            int32_t ih = oh * SH - PH + kh * DH;
                            int32_t id = od * SD - PD + kd * DD;
                            const int trans_y_idx = ((2 * (kh * KW + kw)) * OH + oh) * OW + ow;
                            float transformed_y = ih + trans_data[trans_offset + trans_y_idx];

                            const int trans_x_idx = ((2 * (kh * KW + kw) + 1) * OH + oh) * OW + ow;
                            float transformed_x = iw + trans_data[trans_offset + trans_x_idx];

                            if (transformed_x < 0 || transformed_x >= (int32_t) IW ||
                                transformed_y < 0 || transformed_y >= (int32_t) IH ||
                                id < 0 || id >= (int32_t) ID)
                                continue;

                            auto get_data_index = [&](int h, int w) -> int {
                                return mb * IC * ID * IH * IW
                                       + g * IC / GC * ID * IH * IW
                                       + ic * ID * IH * IW
                                       + id * IH * IW
                                       + h * IW
                                       + w;
                            };

                            size_t widx = g * OC / GC * IC / GC * KD * KH * KW
                                          + oc * IC / GC * KD * KH * KW
                                          + ic * KD * KH * KW
                                          + kd * KH * KW
                                          + kh * KW
                                          + kw;

                            const int top_y_index = floor(transformed_y);
                            const int bottom_y_index = fmin(ceil(transformed_y), IH - 1);
                            const int left_x_index = floor(transformed_x);
                            const int right_x_index = fmin(ceil(transformed_x), IW - 1);

                            const float top_left = src_data[get_data_index(top_y_index, left_x_index)];
                            const float top_right = src_data[get_data_index(top_y_index,
                                                                            right_x_index)];
                            const float bottom_left = src_data[get_data_index(bottom_y_index,
                                                                              left_x_index)];
                            const float bottom_right = src_data[get_data_index(bottom_y_index,
                                                                               right_x_index)];

                            const float top =
                                    top_left + (top_right - top_left) * (transformed_x - left_x_index);
                            const float bottom = bottom_left + (bottom_right - bottom_left) *
                                                               (transformed_x - left_x_index);

                            float val = top + (bottom - top) * (transformed_y - top_y_index);

                            dst_data[oidx] += val * weights_data[widx];
                        }
                    }
                }
            }
        }
    });
}

template<>
void ref_def_conv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                         Blob& dst,
                         const ie_fp16* weights_data,
                         size_t /*weights_size*/,
                         const ie_fp16* bias_data,
                         size_t /*bias_size*/,
                         const CommonTestUtils::def_conv_common_params& prm) {
    if (srcs[0]->getTensorDesc().getLayout() != Layout::NCHW &&
        srcs[0]->getTensorDesc().getLayout() != Layout::NCDHW)
        IE_THROW() << "Reference FP16 deformable convolution supports NCHW and NCDHW layouts only";
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
    size_t IW = src_dims[0];
    size_t IH = src_dims[1];
    size_t ID = src_dims.size() == 5lu ? src_dims[2] : 1lu;
    size_t IC = src_dims.size() == 5lu ? src_dims[3] : src_dims[2];

    auto dst_dims = dst.getTensorDesc().getDims();
    size_t OW = dst_dims[0];
    size_t OH = dst_dims[1];
    size_t OD = dst_dims.size() == 5lu ? dst_dims[2] : 1lu;
    size_t OC = prm.out_c;

    const auto* src_data = srcs[0]->cbuffer().as<const ie_fp16 *>();
    const auto* trans_data = srcs[1]->cbuffer().as<const ie_fp16 *>();
    auto* dst_data = dst.buffer().as<ie_fp16 *>();
    const int channel_per_deformable_group = IC / prm.deformable_group;

    parallel_for4d(GC, OC / GC, OD, OH, [&](size_t g, size_t oc, size_t od, size_t oh) {
        for (uint32_t ow = 0; ow < OW; ow++) {
            size_t oidx = g * OC / GC * OD * OH * OW
                          + oc * OD * OH * OW
                          + od * OH * OW
                          + oh * OW
                          + ow;
            if (bias_data)
                dst_data[oidx] = bias_data[g * OC / GC + oc];

            for (size_t ic = 0; ic < IC / GC; ic++) {
                const int deformable_group_idx = ic / channel_per_deformable_group;
                const int trans_offset = deformable_group_idx * 2 * KH * KW * OW * OW;

                for (size_t kd = 0; kd < KD; kd++) {
                    for (size_t kh = 0; kh < KH; kh++) {
                        for (size_t kw = 0; kw < KW; kw++) {
                            int32_t iw = ow * SW - PW + kw * DW;
                            int32_t ih = oh * SH - PH + kh * DH;
                            int32_t id = od * SD - PD + kd * DD;
                            const int trans_y_idx = ((2 * (kh * KW + kw)) * OW + oh) * OW + ow;
                            float transformed_y = ih + PrecisionUtils::f16tof32(trans_data[trans_offset + trans_y_idx]);

                            const int trans_x_idx = ((2 * (kh * KW + kw) + 1) * OW + oh) * OW + ow;
                            float transformed_x = iw + PrecisionUtils::f16tof32(trans_data[trans_offset + trans_x_idx]);

                            if (transformed_x < 0 || transformed_x >= (int32_t) IW ||
                                transformed_y < 0 || transformed_y >= (int32_t) IH ||
                                id < 0 || id >= (int32_t) ID)
                                continue;

                            auto get_data_index = [&](int h, int w) -> int {
                                return g * IC / GC * ID * IH * IW
                                       + ic * ID * IH * IW
                                       + id * IH * IW
                                       + h * IW
                                       + w;
                            };

                            size_t widx = g * OC / GC * IC / GC * KD * KH * KW
                                          + oc * IC / GC * KD * KH * KW
                                          + ic * KD * KH * KW
                                          + kd * KH * KW
                                          + kh * KW
                                          + kw;

                            const int top_y_index    = floor(transformed_y);
                            const int bottom_y_index = fmin(ceil(transformed_y), IH - 1);
                            const int left_x_index   = floor(transformed_x);
                            const int right_x_index  = fmin(ceil(transformed_x), IW - 1);

                            const float top_left = PrecisionUtils::f16tof32(src_data[get_data_index(top_y_index, left_x_index)]);
                            const float top_right = PrecisionUtils::f16tof32(src_data[get_data_index(top_y_index, right_x_index)]);
                            const float bottom_left = PrecisionUtils::f16tof32(src_data[get_data_index(bottom_y_index, left_x_index)]);
                            const float bottom_right = PrecisionUtils::f16tof32(src_data[get_data_index(bottom_y_index, right_x_index)]);

                            const float top = top_left + (top_right - top_left) * (transformed_x - left_x_index);
                            const float bottom = bottom_left + (bottom_right - bottom_left) * (transformed_x - left_x_index);

                            float val = top + (bottom - top) * (transformed_y - top_y_index);

                            dst_data[oidx] += PrecisionUtils::f32tof16(val * PrecisionUtils::f16tof32(weights_data[widx]));
                        }
                    }
                }
            }
        }
    });
}
