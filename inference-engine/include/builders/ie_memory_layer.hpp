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
 * @brief The class represents a builder for Memory layer
 */
class INFERENCE_ENGINE_API_CLASS(MemoryLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit MemoryLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit MemoryLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit MemoryLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    MemoryLayer& setName(const std::string& name);

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
    MemoryLayer& setOutputPort(const Port& port);
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
    MemoryLayer& setInputPort(const Port& port);
    /**
     * @brief Returns memory ID
     * @return String with memory ID
     */
    const std::string getId() const;
    /**
     * @brief Sets memory ID
     * @param id Memory ID
     * @return reference to layer builder
     */
    MemoryLayer& setId(const std::string& id);
    /**
     * @brief Returns the index of memory layer
     * @return Index
     */
    size_t getIndex() const;
    /**
     * @brief Sets the index of memory layer
     * @param index Index equal 0 means this layer is output one.
     * @return reference to layer builder
     */
    MemoryLayer& setIndex(size_t index);
    /**
     * @brief Returns size of the group
     * @return Size of the group
     */
    size_t getSize() const;
    /**
     * @brief Sets size of the group
     * @param size Size if size equals 2 means this group is a pair (only 2 is supported).
     * @return reference to layer builder
     */
    MemoryLayer& setSize(size_t size);
};

}  // namespace Builder
}  // namespace InferenceEngine
