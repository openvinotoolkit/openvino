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
 * @brief The class represents a builder for Clamp layer
 */
class INFERENCE_ENGINE_API_CLASS(ClampLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ClampLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit ClampLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit ClampLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ClampLayer& setName(const std::string& name);

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
    ClampLayer& setPort(const Port& port);
    /**
     * @brief Returns minimum value
     * @return minimum value
     */
    float getMinValue() const;
    /**
     * @brief Sets minimum value
     * @param minValue Minimum value
     * @return reference to layer builder
     */
    ClampLayer& setMinValue(float minValue);
    /**
     * @brief Returns maximum value
     * @return Maximum value
     */
    float getMaxValue() const;
    /**
     * @brief Sets maximum value
     * @param maxValue Maximum value
     * @return reference to layer builder
     */
    ClampLayer& setMaxValue(float maxValue);
};

}  // namespace Builder
}  // namespace InferenceEngine
