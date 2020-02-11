// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file
 */

#pragma once

#include <builders/ie_convolution_layer.hpp>
#include <ie_network.hpp>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @deprecated Use ngraph API instead.
 * @brief The class represents a builder for Deconvolution layer
 */
IE_SUPPRESS_DEPRECATED_START
class INFERENCE_ENGINE_NN_BUILDER_API_CLASS(DeconvolutionLayer): public ConvolutionLayer {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit DeconvolutionLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit DeconvolutionLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit DeconvolutionLayer(const Layer::CPtr& layer);
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace Builder
}  // namespace InferenceEngine
