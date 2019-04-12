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
 * @brief The class represents a builder for PReLU layer
 */
class INFERENCE_ENGINE_API_CLASS(PReLULayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit PReLULayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit PReLULayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit PReLULayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    PReLULayer& setName(const std::string& name);

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
    PReLULayer& setPort(const Port& port);
    /**
     * @brief Returns channel shared flag
     * @return true if negative slope shared across channels
     */
    bool getChannelShared() const;
    /**
     * @brief Sets channel shared flag
     * @param flag true if negative slope shared across channels
     * @return reference to layer builder
     */
    PReLULayer& setChannelShared(bool flag);
};

}  // namespace Builder
}  // namespace InferenceEngine
