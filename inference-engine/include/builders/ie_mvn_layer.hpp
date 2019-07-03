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
 * @brief The class represents a builder for MVN layer
 */
class INFERENCE_ENGINE_API_CLASS(MVNLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit MVNLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit MVNLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit MVNLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    MVNLayer& setName(const std::string& name);

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
    MVNLayer& setPort(const Port& port);
    /**
     * @brief Returns across channels value
     * @return true if mean values are shared across channels
     */
    bool getAcrossChannels() const;
    /**
     * @brief Sets across channels
     * @param flag true if mean values are shared across channels
     * @return reference to layer builder
     */
    MVNLayer& setAcrossChannels(bool flag);
    /**
     * @brief Returns normalize variance
     * @return true if variance normalization is performed
     */
    bool getNormalize() const;
    /**
     * @brief Sets normalize variance
     * @param flag true if variance normalization is performed
     * @return reference to layer builder
     */
    MVNLayer& setNormalize(bool flag);
    /**
     * @brief Return epsilon
     * @return Epsilon
     */
    float getEpsilon() const;
    /**
     * @brief Sets epsilon
     * @param eps Epsilon
     * @return reference to layer builder
     */
    MVNLayer& setEpsilon(float eps);
};

}  // namespace Builder
}  // namespace InferenceEngine
