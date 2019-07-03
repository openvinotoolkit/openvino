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
 * @brief The class represents a builder for PriorBox layer
 */
class INFERENCE_ENGINE_API_CLASS(PriorBoxLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit PriorBoxLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit PriorBoxLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit PriorBoxLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    PriorBoxLayer& setName(const std::string& name);

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
    PriorBoxLayer& setOutputPort(const Port& port);
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
    PriorBoxLayer& setInputPorts(const std::vector<Port>& ports);
    /**
     * @brief Returns the minimum box size in pixels
     * @return Minimum box size
     */
    size_t getMinSize() const;
    /**
     * @brief Sets the minimum box size in pixels
     * @param minSize Minimum size
     * @return reference to layer builder
     */
    PriorBoxLayer& setMinSize(size_t minSize);
    /**
     * @brief Returns the maximum box size in pixels
     * @return maximum size
     */
    size_t getMaxSize() const;
    /**
     * @brief Sets the maximum box size in pixels
     * @param maxSize Maximum size
     * @return reference to layer builder
     */
    PriorBoxLayer& setMaxSize(size_t maxSize);
    /**
     * @brief Returns a distance between box centers
     * @return Distance
     */
    float getStep() const;
    /**
     * @brief Sets a distance between box centers
     * @param step Distance
     * @return reference to layer builder
     */
    PriorBoxLayer& setStep(float step);
    /**
     * @brief Returns a shift of box respectively to top left corner
     * @return Shift
     */
    float getOffset() const;
    /**
     * @brief Sets a shift of box respectively to top left corner
     * @param offset Shift
     * @return reference to layer builder
     */
    PriorBoxLayer& setOffset(float offset);
    /**
     * @brief Returns a variance of adjusting bounding boxes
     * @return Variance
     */
    float getVariance() const;
    /**
     * @brief Sets a variance of adjusting bounding boxes
     * @param variance Variance
     * @return reference to layer builder
     */
    PriorBoxLayer& setVariance(float variance);
    /**
     * @brief Returns a flag that denotes type of inference
     * @return true if max_size is used
     */
    bool getScaleAllSizes() const;
    /**
     * @brief Sets a flag that denotes a type of inference
     * @param flag max_size is used if true
     * @return reference to layer builder
     */
    PriorBoxLayer& setScaleAllSizes(bool flag);
    /**
     * @brief Returns clip flag
     * @return true if each value in the output blob is within [0,1]
     */
    bool getClip() const;
    /**
     * @brief sets clip flag
     * @param flag true if each value in the output blob is within [0,1]
     * @return reference to layer builder
     */
    PriorBoxLayer& setClip(bool flag);
    /**
     * @brief Returns flip flag
     * @return list of boxes is augmented with the flipped ones if true
     */
    bool getFlip() const;
    /**
     * @brief Sets flip flag
     * @param flag true if list of boxes is augmented with the flipped ones
     * @return reference to layer builder
     */
    PriorBoxLayer& setFlip(bool flag);
    /**
     * @brief Returns a variance of aspect ratios
     * @return Vector of aspect ratios
     */
    const std::vector<size_t> getAspectRatio() const;
    /**
     * @brief Sets a variance of aspect ratios
     * @param aspectRatio Vector of aspect ratios
     * @return reference to layer builder
     */
    PriorBoxLayer& setAspectRatio(const std::vector<size_t>& aspectRatio);
};

}  // namespace Builder
}  // namespace InferenceEngine
