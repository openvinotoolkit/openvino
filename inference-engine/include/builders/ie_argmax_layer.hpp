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
 * @brief The class represents a builder for ArgMax layer
 */
class INFERENCE_ENGINE_API_CLASS(ArgMaxLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ArgMaxLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit ArgMaxLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit ArgMaxLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ArgMaxLayer& setName(const std::string& name);

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
    ArgMaxLayer& setPort(const Port& port);
    /**
     * @brief Returns axis
     * @return Axis
     */
    int getAxis() const;
    /**
     * @brief Sets axis
     * @param axis Axis
     * @return reference to layer builder
     */
    ArgMaxLayer& setAxis(int axis);
    /**
     * @brief Returns top K
     * @return Top K
     */
    size_t getTopK() const;
    /**
     * @brief Sets top K
     * @param topK Top K
     * @return reference to layer builder
     */
    ArgMaxLayer& setTopK(size_t topK);
    /**
     * @brief Returns output maximum value
     * @return Output maximum value
     */
    size_t getOutMaxVal() const;
    /**
     * @brief Sets output maximum value
     * @param size Maximum value
     * @return reference to layer builder
     */
    ArgMaxLayer& setOutMaxVal(size_t size);
};

}  // namespace Builder
}  // namespace InferenceEngine
