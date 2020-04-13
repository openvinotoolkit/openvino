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
class INFERENCE_ENGINE_NN_BUILDER_API_CLASS(DeformableConvolutionLayer): public ConvolutionLayer {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit DeformableConvolutionLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit DeformableConvolutionLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit DeformableConvolutionLayer(const Layer::CPtr& layer);
    /**
     * @brief Return deformable_group size
     * @return Deformable group size
     */
    size_t getDeformableGroup() const;
    /**
     * @brief Sets deformable group size
     * @param deformableGroup Deformable group
     * @return reference to layer builder
     */
    Builder::DeformableConvolutionLayer& setDeformableGroup(size_t deformableGroup);
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace Builder
}  // namespace InferenceEngine
