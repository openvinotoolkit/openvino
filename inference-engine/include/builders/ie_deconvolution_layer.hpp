// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_convolution_layer.hpp>
#include <ie_inetwork.hpp>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for Deconvolution layer
 */
class INFERENCE_ENGINE_API_CLASS(DeconvolutionLayer): public ConvolutionLayer {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit DeconvolutionLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param genLayer generic builder
     */
    explicit DeconvolutionLayer(Layer& genLayer);
};

}  // namespace Builder
}  // namespace InferenceEngine
