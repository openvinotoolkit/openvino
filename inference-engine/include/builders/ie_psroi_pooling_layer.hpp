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
 * @brief The class represents a builder for PSROIPooling layer
 */
class INFERENCE_ENGINE_API_CLASS(PSROIPoolingLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit PSROIPoolingLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit PSROIPoolingLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit PSROIPoolingLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    PSROIPoolingLayer& setName(const std::string& name);

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
    PSROIPoolingLayer& setInputPorts(const std::vector<Port>& ports);
    /**
     * @brief Returns output ports
     * @return Vector of output ports
     */
    const Port& getOutputPort() const;
    /**
     * @brief Sets output ports
     * @param port Vector of output ports
     * @return reference to layer builder
     */
    PSROIPoolingLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns multiplicative spatial scale factor to translate ROI coordinates
     * @return Spatial scale factor
     */
    float getSpatialScale() const;
    /**
     * @brief Sets multiplicative spatial scale factor to translate ROI coordinates
     * @param spatialScale Spatial scale factor
     * @return reference to layer builder
     */
    PSROIPoolingLayer& setSpatialScale(float spatialScale);
    /**
     * @brief Returns pooled output channel number
     * @return Output channel number
     */
    size_t getOutputDim() const;
    /**
     * @brief Sets pooled output channel number
     * @param outDim Output channel number
     * @return reference to layer builder
     */
    PSROIPoolingLayer& setOutputDim(size_t outDim);
    /**
     * @brief Returns number of groups to encode position-sensitive score maps
     * @return Number of groups
     */
    size_t getGroupSize() const;
    /**
     * @brief Sets number of groups to encode position-sensitive score maps
     * @param size Number of groups
     * @return reference to layer builder
     */
    PSROIPoolingLayer& setGroupSize(size_t size);
};

}  // namespace Builder
}  // namespace InferenceEngine



