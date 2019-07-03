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
 * @brief The class represents a builder for Split layer
 */
class INFERENCE_ENGINE_API_CLASS(SplitLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit SplitLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit SplitLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit SplitLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    SplitLayer& setName(const std::string& name);

    /**
     * @brief Returns output ports
     * @return Vector of output ports
     */
    const std::vector<Port>& getOutputPorts() const;
    /**
     * @brief Sets output ports
     * @param ports Vector of output ports
     * @return reference to layer builder
     */
    SplitLayer& setOutputPorts(const std::vector<Port>& ports);
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
    SplitLayer& setInputPort(const Port& port);
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
    SplitLayer& setAxis(size_t axis);
};

}  // namespace Builder
}  // namespace InferenceEngine
