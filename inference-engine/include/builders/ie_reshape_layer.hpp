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
 * @brief The class represents a builder for Reshape layer
 */
class INFERENCE_ENGINE_API_CLASS(ReshapeLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ReshapeLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit ReshapeLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit ReshapeLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ReshapeLayer& setName(const std::string& name);

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
    ReshapeLayer& setInputPort(const Port& port);
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
    ReshapeLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns reshape dimensions
     * @return Dimensions
     */
    const std::vector<int> getDims() const;
    /**
     * @brief Sets reshape dimensions
     * @param dims Dimensions
     * @return reference to layer builder
     */
    ReshapeLayer& setDims(const std::vector<int>& dims);
};

}  // namespace Builder
}  // namespace InferenceEngine
