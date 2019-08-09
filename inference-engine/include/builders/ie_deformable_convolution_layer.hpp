// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_convolution_layer.hpp>
#include <ie_network.hpp>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for Deconvolution layer
 */
class INFERENCE_ENGINE_API_CLASS(DeformableConvolutionLayer): public ConvolutionLayer {
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

}  // namespace Builder
}  // namespace InferenceEngine
