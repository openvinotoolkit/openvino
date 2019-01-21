// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_deconvolution_layer.hpp>
#include <details/caseless.hpp>
#include <string>

using namespace InferenceEngine;

Builder::DeconvolutionLayer::DeconvolutionLayer(const std::string& name): ConvolutionLayer(name) {
    getLayer().setType("Deconvolution");
}
Builder::DeconvolutionLayer::DeconvolutionLayer(Layer& genLayer): ConvolutionLayer(genLayer.getName()) {
    getLayer().setName("");
    getLayer().setType("");
    getLayer() = genLayer;
    if (!details::CaselessEq<std::string>()(getLayer().getType(), "Deconvolution"))
        THROW_IE_EXCEPTION << "Cannot create DeconvolutionLayer decorator for layer " << getLayer().getType();
}
