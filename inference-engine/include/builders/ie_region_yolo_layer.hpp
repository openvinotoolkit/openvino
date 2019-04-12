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
 * @brief The class represents a builder for RegionYolo layer
 */
class INFERENCE_ENGINE_API_CLASS(RegionYoloLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit RegionYoloLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit RegionYoloLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit RegionYoloLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    RegionYoloLayer& setName(const std::string& name);

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
    RegionYoloLayer& setInputPort(const Port& port);
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
    RegionYoloLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns number of coordinates for each region
     * @return Number of coordinates
     */
    int getCoords() const;
    /**
     * @brief Sets number of coordinates for each region
     * @param coords Number of coordinates
     * @return reference to layer builder
     */
    RegionYoloLayer& setCoords(int coords);
    /**
     * @brief Returns number of classes for each region
     * @return Number of classes
     */
    int getClasses() const;
    /**
     * @brief Sets number of classes for each region
     * @param classes number of classes
     * @return reference to layer builder
     */
    RegionYoloLayer& setClasses(int classes);
    /**
     * @brief Returns number of regions
     * @return Number of regions
     */
    int getNum() const;
    /**
     * @brief Sets number of regions
     * @param num Number of regions
     * @return reference to layer builder
     */
    RegionYoloLayer& setNum(int num);
    /**
     * @brief Returns a flag which specifies the method of infer
     * @return true if softmax is performed
     */
    bool getDoSoftMax() const;
    /**
     * @brief Sets a flag which specifies the method of infer
     * @param flag softmax is performed if true
     * @return reference to layer builder
     */
    RegionYoloLayer& setDoSoftMax(bool flag);
    /**
     * @brief Returns anchors coordinates of regions
     * @return anchors coordinates
     */
    float getAnchors() const;
    /**
     * @brief Sets anchors coordinates of regions
     * @param anchors Anchors coordinates
     * @return reference to layer builder
     */
    RegionYoloLayer& setAnchors(float anchors);
    /**
     * @brief Returns mask
     * @return Mask
     */
    int getMask() const;
    /**
     * @brief Sets mask
     * @param mask Specifies which anchors to use
     * @return reference to layer builder
     */
    RegionYoloLayer& setMask(int mask);
    /**
     * @brief Returns the number of the dimension from which flattening is performed
     * @return Axis
     */
    size_t getAxis() const;
    /**
     * @brief Sets the number of the dimension from which flattening is performed
     * @param axis Axis
     * @return reference to layer builder
     */
    RegionYoloLayer& setAxis(size_t axis);
    /**
     * @brief Returns the number of the dimension on which flattening is ended
     * @return End axis
     */
    size_t getEndAxis() const;
    /**
     * @brief Sets the number of the dimension on which flattening is ended
     * @param axis End axis
     * @return reference to layer builder
     */
    RegionYoloLayer& setEndAxis(size_t axis);
};

}  // namespace Builder
}  // namespace InferenceEngine





