// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_decorator.hpp>
#include <ie_network.hpp>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for Concat layer
 */
class INFERENCE_ENGINE_API_CLASS(ConcatLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ConcatLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit ConcatLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit ConcatLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ConcatLayer& setName(const std::string& name);

    /**
     * @brief Returns vector with input ports
     * @return vector with ports
     */
    const std::vector<Port>& getInputPorts() const;
    /**
     * @brief Sets input ports
     * @param ports Vector of input ports
     * @return reference to layer builder
     */
    ConcatLayer& setInputPorts(const std::vector<Port>& ports);
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
    ConcatLayer& setOutputPort(const Port& port);
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
    ConcatLayer& setAxis(size_t axis);
};

}  // namespace Builder
}  // namespace InferenceEngine
