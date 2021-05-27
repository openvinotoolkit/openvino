// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>
#include <legacy/ie_layers_property.hpp>
#include <ie_blob.h>
#include <precision_utils.h>
#include <legacy/ie_layers_internal.hpp>
#include "common_layers_params.hpp"

template<typename data_t>
void ref_def_conv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                     InferenceEngine::Blob& dst,
                     const data_t* weights_data,
                     size_t weights_size,
                     const data_t* bias_data,
                     size_t bias_size,
                     const CommonTestUtils::def_conv_common_params& prm);

void DeformableConvolution_parseParams(InferenceEngine::CNNLayer* layer);

template<typename data_t>
void common_ref_def_convolution_wrap(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                                 InferenceEngine::Blob::Ptr& dst,
                                 const data_t* weights_data,
                                 size_t weights_size,
                                 const data_t* bias_data,
                                 size_t bias_size,
                                 const std::map<std::string, std::string>& params_map) {
    InferenceEngine::LayerParams lp{};
    InferenceEngine::ConvolutionLayer convLayer(lp);
    auto data = std::make_shared<InferenceEngine::Data>("insData", srcs[0]->getTensorDesc());
    convLayer.params = params_map;
    convLayer.insData.push_back(data);
    DeformableConvolution_parseParams(&convLayer);

    CommonTestUtils::conv_common_params params;
    params.kernel = convLayer._kernel;
    auto allPad = InferenceEngine::getPaddings(convLayer);
    params.pads_begin = allPad.begin;
    params.pads_end = allPad.end;
    params.stride = convLayer._stride;
    params.dilation = convLayer._dilation;
    params.out_c = convLayer._out_depth;
    params.group = convLayer._group;

    ref_def_conv_common<data_t>(srcs,
                            *dst.get(),
                            weights_data,
                            weights_size,
                            bias_data,
                            bias_size,
                            params);
}
