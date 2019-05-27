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
 * @brief The class represents a builder for Eltwise layer
 */
class INFERENCE_ENGINE_API_CLASS(EltwiseLayer): public LayerDecorator {
public:
    /**
     * @brief The enum defines all Eltwise types
     */
    enum EltwiseType {
        SUM = 1,
        MAX,
        MUL,
        SUB,
        DIV,
        MIN,
        SQUARED_DIFF
    };

    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit EltwiseLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit EltwiseLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit EltwiseLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    EltwiseLayer& setName(const std::string& name);

    /**
     * @brief Returns input ports
     * @return Vector of input ports
     */
    const std::vector<Port>& getInputPorts() const;
    /**
     * @brief Sets input ports
     * @param ports Vector of input ports
     * @return reference to layer builder
     */
    EltwiseLayer& setInputPorts(const std::vector<Port>& ports);
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
    EltwiseLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns eltwise type
     * @return Eltwise type
     */
    EltwiseType getEltwiseType() const;
    /**
     * @brief Sets eltwise type
     * @param type Eltwise type
     * @return reference to layer builder
     */
    EltwiseLayer& setEltwiseType(EltwiseType type);
    /**
     * @brief Returns eltwise scales
     * @return Vector of scales
     */
    const std::vector<float> getScales() const;
    /**
     * @brief Sets eltwise scales
     * @param scales Vector of scales
     * @return reference to layer builder
     */
    EltwiseLayer& setScales(const std::vector<float>& scales);

private:
    EltwiseType type = SUM;
};

}  // namespace Builder
}  // namespace InferenceEngine
