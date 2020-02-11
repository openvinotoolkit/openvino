// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file
 */

#pragma once

#include <builders/ie_layer_decorator.hpp>
#include <ie_network.hpp>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Builder {

/**
 * @deprecated Use ngraph API instead.
 * @brief The class represents a builder for ReorgYolo layer
 */
IE_SUPPRESS_DEPRECATED_START
class INFERENCE_ENGINE_NN_BUILDER_API_CLASS(ReorgYoloLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ReorgYoloLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit ReorgYoloLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer const pointer to generic builder
     */
    explicit ReorgYoloLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ReorgYoloLayer& setName(const std::string& name);

    /**
     * @brief Returns input port
     * @return Input port
     */
    const Port& getInputPort() const;
    /**
     * @brief Sets input port
     * @param ports Input port
     * @return reference to layer builder
     */
    ReorgYoloLayer& setInputPort(const Port& ports);
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
    ReorgYoloLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns distance of cut throws in output blobs
     * @return Stride
     */
    int getStride() const;
    /**
     * @brief Sets distance of cut throws in output blobs
     * @param stride Stride
     * @return reference to layer builder
     */
    ReorgYoloLayer& setStride(int stride);
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace Builder
}  // namespace InferenceEngine
