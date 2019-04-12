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
 * @brief The class represents a builder for Proposal layer
 */
class INFERENCE_ENGINE_API_CLASS(ProposalLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ProposalLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit ProposalLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit ProposalLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ProposalLayer& setName(const std::string& name);

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
    ProposalLayer& setOutputPort(const Port& port);
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
    ProposalLayer& setInputPorts(const std::vector<Port>& ports);
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
    ProposalLayer& setPostNMSTopN(size_t topN);
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
    ProposalLayer& setPreNMSTopN(size_t topN);
    /**
     * @brief Returns minimum value of the proposal to be taken into consideration
     * @return Threshold
     */
    float getNMSThresh() const;
    /**
     * @brief Sets minimum value of the proposal to be taken into consideration
     * @param thresh Threshold
     * @return reference to layer builder
     */
    ProposalLayer& setNMSThresh(float thresh);
    /**
     * @brief Returns base size for anchor generation
     * @return Base size
     */
    size_t getBaseSize() const;
    /**
     * @brief Sets base size for anchor generation
     * @param baseSize Base size for anchor generation
     * @return reference to layer builder
     */
    ProposalLayer& setBaseSize(size_t baseSize);
    /**
     * @brief Returns minimum size of box to be taken into consideration
     * @return Minimum size
     */
    size_t getMinSize() const;
    /**
     * @brief Sets minimum size of box to be taken into consideration
     * @param minSize Minimum size of the box
     * @return reference to layer builder
     */
    ProposalLayer& setMinSize(size_t minSize);
    /**
     * @brief Returns step size to slide over boxes in pixels
     * @return Step size
     */
    size_t getFeatStride() const;
    /**
     * @brief Sets step size to slide over boxes in pixels
     * @param featStride Step size
     * @return reference to layer builder
     */
    ProposalLayer& setFeatStride(size_t featStride);
    /**
     * @brief Returns scales for anchor generation
     * @return Vector of scales
     */
    const std::vector<float> getScale() const;
    /**
     * @brief Sets scales for anchor generation
     * @param scales Vector of scales
     * @return reference to layer builder
     */
    ProposalLayer& setScale(const std::vector<float>& scales);
    /**
     * @brief Returns ratios for anchor generation
     * @return Vector of ratios
     */
    const std::vector<float> getRatio() const;
    /**
     * @brief Sets ratios for anchor generation
     * @param ratios Vector of scales
     * @return reference to layer builder
     */
    ProposalLayer& setRatio(const std::vector<float>& ratios);
};

}  // namespace Builder
}  // namespace InferenceEngine

