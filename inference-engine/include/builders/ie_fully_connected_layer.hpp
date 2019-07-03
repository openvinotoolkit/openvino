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
 * @brief The class represents a builder for FullyConnected layer
 */
class INFERENCE_ENGINE_API_CLASS(FullyConnectedLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit FullyConnectedLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit FullyConnectedLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit FullyConnectedLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    FullyConnectedLayer& setName(const std::string& name);

    /**
     * @brief Returns input port
     * @return Input port
     */
    const Port& getInputPort() const;
    /**
     * @brief Sets input port
     * @param port Input port
     * @return reference to layer builder
     */
    FullyConnectedLayer& setInputPort(const Port& port);
    /**
     * @brief Returns output port
     * @return Output port
     */
    const Port& getOutputPort() const;
    /**
     * @brief Sets output port
     * @param port Output port
     * @return reference to layer builder
     */
    FullyConnectedLayer& setOutputPort(const Port& port);
    /**
     * @brief Return output size
     * @return Output size
     */
    size_t getOutputNum() const;
    /**
     * @brief Sets output size
     * @param outNum Output size
     * @return reference to layer builder
     */
    FullyConnectedLayer& setOutputNum(size_t outNum);
};

}  // namespace Builder
}  // namespace InferenceEngine
