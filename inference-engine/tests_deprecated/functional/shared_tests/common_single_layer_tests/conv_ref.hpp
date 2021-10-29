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

template<typename wei_data_t, typename bias_data_t>
void ref_conv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                     InferenceEngine::Blob& dst,
                     const wei_data_t* weights_data,
                     size_t weights_size,
                     const bias_data_t* bias_data,
                     size_t bias_size,
                     const CommonTestUtils::conv_common_params& prm);


void Convolution_parseParams(InferenceEngine::CNNLayer* layer);

template<typename wei_data_t, typename bias_data_t>
void common_ref_convolution_wrap(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                                 InferenceEngine::Blob::Ptr& dst,
                                 const wei_data_t* weights_data,
                                 size_t weights_size,
                                 const bias_data_t* bias_data,
                                 size_t bias_size,
                                 const std::map<std::string, std::string>& params_map) {
    InferenceEngine::LayerParams lp{};
    InferenceEngine::ConvolutionLayer convLayer(lp);
    auto data = std::make_shared<InferenceEngine::Data>("insData", srcs[0]->getTensorDesc());
    convLayer.params = params_map;
    convLayer.insData.push_back(data);
    Convolution_parseParams(&convLayer);

    CommonTestUtils::conv_common_params params;
    params.kernel = convLayer._kernel;
    auto allPad = InferenceEngine::getPaddings(convLayer);
    params.pads_begin = allPad.begin;
    params.pads_end = allPad.end;
    params.stride = convLayer._stride;
    params.dilation = convLayer._dilation;
    params.out_c = convLayer._out_depth;
    params.group = convLayer._group;

    ref_conv_common<>(srcs,
                      *dst.get(),
                      weights_data,
                      weights_size,
                      bias_data,
                      bias_size,
                      params);
}
