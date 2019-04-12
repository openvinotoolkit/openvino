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
 * @brief The class represents a builder for SimplerNMS layer
 */
class INFERENCE_ENGINE_API_CLASS(SimplerNMSLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit SimplerNMSLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit SimplerNMSLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit SimplerNMSLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    SimplerNMSLayer& setName(const std::string& name);

    /**
     * @brief Returns input ports
     * @return Vector of input ports
     */
    const std::vector<Port>& getInputPorts() const;
    /**
     * @brief Sets input ports
     * @param ports Vector of input ports
     */
    SimplerNMSLayer& setInputPorts(const std::vector<Port>& ports);
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
    SimplerNMSLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns the quantity of bounding boxes before applying NMS
     * @return Quantity of bounding boxes
     */
    size_t getPreNMSTopN() const;
    /**
     * @brief Sets the quantity of bounding boxes before applying NMS
     * @param topN Quantity of bounding boxes
     * @return reference to layer builder
     */
    SimplerNMSLayer& setPreNMSTopN(size_t topN);
    /**
     * @brief Returns the quantity of bounding boxes after applying NMS
     * @return Quantity of bounding boxes
     */
    size_t getPostNMSTopN() const;
    /**
     * @brief Sets the quantity of bounding boxes after applying NMS
     * @param topN Quantity of bounding boxes
     * @return reference to layer builder
     */
    SimplerNMSLayer& setPostNMSTopN(size_t topN);
    /**
     * @brief Returns the step size to slide over boxes in pixels
     * @return Step size
     */
    size_t getFeatStride() const;
    /**
     * @brief Sets the step size to slide over boxes in pixels
     * @param featStride Step size
     * @return reference to layer builder
     */
    SimplerNMSLayer& setFeatStride(size_t featStride);
    /**
     * @brief Returns the minimum size of box to be taken into consideration
     * @return Minimum size
     */
    size_t getMinBoxSize() const;
    /**
     * @brief Sets the minimum size of box to be taken into consideration
     * @param minSize Minimum size
     * @return reference to layer builder
     */
    SimplerNMSLayer& setMinBoxSize(size_t minSize);
    /**
     * @brief Returns scale for anchor boxes generating
     * @return Scale for anchor boxes
     */
    size_t getScale() const;
    /**
     * @brief Sets scale for anchor boxes generating
     * @param scale Scale for anchor boxes
     * @return reference to layer builder
     */
    SimplerNMSLayer& setScale(size_t scale);

    /**
     * @brief Returns the minimum value of the proposal to be taken into consideration
     * @return Threshold
     */
    float getCLSThreshold() const;
    /**
     * @brief Sets the minimum value of the proposal to be taken into consideration
     * @param threshold Minimum value
     * @return reference to layer builder
     */
    SimplerNMSLayer& setCLSThreshold(float threshold);
    /**
     * @brief Returns the minimum ratio of boxes overlapping to be taken into consideration
     * @return Threshold
     */
    float getIOUThreshold() const;
    /**
     * @brief Sets the minimum ratio of boxes overlapping to be taken into consideration
     * @param threshold Minimum value
     * @return reference to layer builder
     */
    SimplerNMSLayer& setIOUThreshold(float threshold);
};

}  // namespace Builder
}  // namespace InferenceEngine

