// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_decorator.hpp>
#include <ie_network.hpp>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for SoftMax layer
 */
class INFERENCE_ENGINE_API_CLASS(SoftMaxLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit SoftMaxLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit SoftMaxLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit SoftMaxLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    SoftMaxLayer& setName(const std::string& name);

    /**
     * @brief Returns port with shapes for the layer
     * @return Port with shapes
     */
    const Port& getPort() const;
    /**
     * @brief Sets port shapes for the layer
     * @param port Port with shapes
     * @return reference to layer builder
     */
    SoftMaxLayer& setPort(const Port& port);
    /**
     * @brief Returns axis
     * @return Axis
     */
    size_t getAxis() const;
    /**
     * @brief Sets axis
     * @param axis Axis
     * @return reference to layer builder
     */
    SoftMaxLayer& setAxis(size_t axis);
};

}  // namespace Builder
}  // namespace InferenceEngine
