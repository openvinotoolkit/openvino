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
void ref_deconv_common(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                       InferenceEngine::Blob& dst,
                       const data_t* weights_data,
                       size_t weights_size,
                       const data_t* bias_data,
                       size_t bias_size,
                       const CommonTestUtils::conv_common_params& prm);

void Convolution_parseParams(InferenceEngine::CNNLayer* layer);

template<typename data_t>
void common_ref_deconvolution_wrap(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                                 InferenceEngine::Blob::Ptr& dst,
                                 const data_t* weights_data,
                                 size_t weights_size,
                                 const data_t* bias_data,
                                 size_t bias_size,
                                 const std::map<std::string, std::string>& params_map) {
    InferenceEngine::LayerParams lp{};
    InferenceEngine::ConvolutionLayer deconvLayer(lp);
    auto data = std::make_shared<InferenceEngine::Data>("insData", srcs[0]->getTensorDesc());
    deconvLayer.params = params_map;
    deconvLayer.insData.push_back(data);
    Convolution_parseParams(&deconvLayer);

    CommonTestUtils::conv_common_params params;
    params.kernel = deconvLayer._kernel;
    auto allPad = InferenceEngine::getPaddings(deconvLayer);
    params.pads_begin = allPad.begin;
    params.pads_end = allPad.end;
    params.stride = deconvLayer._stride;
    params.dilation = deconvLayer._dilation;
    params.out_c = deconvLayer._out_depth;
    params.group = deconvLayer._group;

    ref_deconv_common<data_t>(srcs,
                            *dst.get(),
                            weights_data,
                            weights_size,
                            bias_data,
                            bias_size,
                            params);
}
