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
 * @brief The class represents a builder for Power layer
 */
class INFERENCE_ENGINE_API_CLASS(PowerLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit PowerLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit PowerLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit PowerLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    PowerLayer& setName(const std::string& name);

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
    PowerLayer& setPort(const Port& port);
    /**
     * @brief Returns power
     * @return Power parameter
     */
    float getPower() const;
    /**
     * @brief Sets the power parameter
     * @param power Power parameter
     * @return reference to layer builder
     */
    PowerLayer& setPower(float power);
    /**
     * @brief Returns scaling parameter
     * @return Scaling
     */
    float getScale() const;
    /**
     * @brief Sets scaling parameter
     * @param scale Scaling parameter
     * @return reference to layer builder
     */
    PowerLayer& setScale(float scale);
    /**
     * @brief Returns shifting parameter
     * @return Shift
     */
    float getShift() const;
    /**
     * @brief Sets shift for the layer
     * @param shift Shifting parameter
     * @return reference to layer builder
     */
    PowerLayer& setShift(float shift);
};

}  // namespace Builder
}  // namespace InferenceEngine
