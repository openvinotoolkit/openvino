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
 * @brief The class represents a builder for ROIPooling layer
 */
class INFERENCE_ENGINE_API_CLASS(ROIPoolingLayer): public LayerFragment {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ROIPoolingLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param genLayer generic builder
     */
    explicit ROIPoolingLayer(Layer& genLayer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ROIPoolingLayer& setName(const std::string& name);

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
    ROIPoolingLayer& setInputPorts(const std::vector<Port>& ports);
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
    ROIPoolingLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns a ratio of the input feature map over the input image size
     * @return Spatial scale
     */
    float getSpatialScale() const;
    /**
     * @brief Sets a ratio of the input feature map over the input image size
     * @param spatialScale Spatial scale
     * @return reference to layer builder
     */
    ROIPoolingLayer& setSpatialScale(float spatialScale);
    /**
     * @brief Returns height and width of the ROI output feature map
     * @return Vector contains height and width
     */
    const std::vector<int> getPooled() const;
    /**
     * @brief Sets height and width of the ROI output feature map
     * @param pooled Vector with height and width
     * @return reference to layer builder
     */
    ROIPoolingLayer& setPooled(const std::vector<int>& pooled);
};

}  // namespace Builder
}  // namespace InferenceEngine
