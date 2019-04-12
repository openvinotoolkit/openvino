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
 * @brief The class represents a builder for PriorBoxClustered layer
 */
class INFERENCE_ENGINE_API_CLASS(PriorBoxClusteredLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit PriorBoxClusteredLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit PriorBoxClusteredLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit PriorBoxClusteredLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    PriorBoxClusteredLayer& setName(const std::string& name);

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
    PriorBoxClusteredLayer& setOutputPort(const Port& port);
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
    PriorBoxClusteredLayer& setInputPorts(const std::vector<Port>& port);
    /**
     * @brief Returns height and width of input image
     * @return input image sizes
     */
    const std::vector<float> getImgSizes() const;
    /**
     * @brief Sets height and width sizes
     * @param sizes Height and width sizes
     * @return reference to layer builder
     */
    PriorBoxClusteredLayer& setImgSizes(const std::vector<float> sizes);
    /**
     * @brief returns distances between (height and width) box centers
     * @return distances
     */
    const std::vector<float> getSteps() const;
    /**
     * @brief Sets distances between box centers for height and width
     * @param steps Distances between box centers
     * @return reference to layer builder
     */
    PriorBoxClusteredLayer& setSteps(const std::vector<float> steps);
    /**
     * @brief returns a distance between box centers
     * @return distance
     */
    float getStep() const;
    /**
     * @brief Sets a distance between box centers
     * @param steps A distance between box centers
     * @return reference to layer builder
     */
    PriorBoxClusteredLayer& setStep(float step);
    /**
     * @brief Returns shift of box respectively to top left corner
     * @return Shift
     */
    float getOffset() const;
    /**
     * @brief Sets shift of box respectively to top left corner
     * @param offset Shift
     * @return reference to layer builder
     */
    PriorBoxClusteredLayer& setOffset(float offset);
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
    PriorBoxClusteredLayer& setVariance(float variance);
    /**
     * @brief Returns desired boxes width in pixels
     * @return width of desired boxes
     */
    float getWidth() const;
    /**
     * @brief Sets desired boxes width in pixels
     * @param width Width of desired boxes
     * @return reference to layer builder
     */
    PriorBoxClusteredLayer& setWidth(float width);
    /**
     * @brief Returns desired boxes height in pixels
     * @return height of desired boxes
     */
    float getHeight() const;
    /**
     * @brief Sets desired boxes height in pixels
     * @param height Height of desired boxes
     * @return reference to layer builder
     */
    PriorBoxClusteredLayer& setHeight(float height);
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
    PriorBoxClusteredLayer& setClip(bool flag);
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
    PriorBoxClusteredLayer& setFlip(bool flag);
};

}  // namespace Builder
}  // namespace InferenceEngine
