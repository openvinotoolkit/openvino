// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_fragment.hpp>
#include <ie_inetwork.hpp>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for Crop layer
 */
class INFERENCE_ENGINE_API_CLASS(CropLayer): public LayerFragment {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit CropLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param genLayer generic builder
     */
    explicit CropLayer(Layer& genLayer);
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

    /**
     * @brief Validates layer before creation
     * @param layer generic layer builder
     */
    static void validate(const Layer& layer);
};

}  // namespace Builder
}  // namespace InferenceEngine
