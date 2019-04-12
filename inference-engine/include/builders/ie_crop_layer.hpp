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
 * @brief The class represents a builder for Crop layer
 */
class INFERENCE_ENGINE_API_CLASS(CropLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit CropLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit CropLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit CropLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    CropLayer& setName(const std::string& name);

    /**
     * @brief Returns input ports
     * @return Vector of input ports
     */
    const std::vector<Port>& getInputPorts() const;
    /**
     * @brief Sets input ports
     * @param port Vector of input ports
     * @return reference to layer builder
     */
    CropLayer& setInputPorts(const std::vector<Port>& ports);
    /**
     * @brief Return output port
     * @return Output port
     */
    const Port& getOutputPort() const;
    /**
     * @brief Sets output port
     * @param port Output port
     * @return reference to layer builder
     */
    CropLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns axis
     * @return Vector of axis
     */
    const std::vector<size_t> getAxis() const;
    /**
     * @brief Sets axis
     * @param axis Vector of axis
     * @return reference to layer builder
     */
    CropLayer& setAxis(const std::vector<size_t>& axis);
    /**
     * @brief Returns offsets
     * @return Vector of offsets
     */
    const std::vector<size_t> getOffset() const;
    /**
     * @brief Sets offsets
     * @param offsets Vector of offsets
     * @return reference to layer builder
     */
    CropLayer& setOffset(const std::vector<size_t>& offsets);
};

}  // namespace Builder
}  // namespace InferenceEngine
