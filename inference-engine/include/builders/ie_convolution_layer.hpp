// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_decorator.hpp>
#include <ie_network.hpp>
#include <vector>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for ArgMax layer
 */
class INFERENCE_ENGINE_API_CLASS(ConvolutionLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ConvolutionLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit ConvolutionLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit ConvolutionLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ConvolutionLayer& setName(const std::string& name);

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
    ConvolutionLayer& setInputPort(const Port& port);
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
    ConvolutionLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns kernel size
     * @return Kernel size
     */
    const std::vector<size_t> getKernel() const;
    /**
     * @brief Sets kernel size
     * @param kernel Kernel size
     * @return reference to layer builder
     */
    ConvolutionLayer& setKernel(const std::vector<size_t>& kernel);
    /**
     * @brief Returns vector of strides
     * @return vector of strides
     */
    const std::vector<size_t> getStrides() const;
    /**
     * @brief Sets strides
     * @param strides vector of strides
     * @return reference to layer builder
     */
    ConvolutionLayer& setStrides(const std::vector<size_t>& strides);
    /**
     * @brief Returns dilations
     * @return vector of dilations
     */
    const std::vector<size_t> getDilation() const;
    /**
     * @brief Sets dilations
     * @param dilation Vector of dilations
     * @return reference to layer builder
     */
    ConvolutionLayer& setDilation(const std::vector<size_t>& dilation);
    /**
     * @brief Returns begin paddings
     * @return vector of paddings
     */
    const std::vector<size_t> getPaddingsBegin() const;
    /**
     * @brief Sets begin paddings
     * @param paddings Vector of paddings
     * @return reference to layer builder
     */
    ConvolutionLayer& setPaddingsBegin(const std::vector<size_t>& paddings);
    /**
     * @brief Return end paddings
     * @return Vector of paddings
     */
    const std::vector<size_t> getPaddingsEnd() const;
    /**
     * @brief Sets end paddings
     * @param paddings Vector of paddings
     * @return reference to layer builder
     */
    ConvolutionLayer& setPaddingsEnd(const std::vector<size_t>& paddings);
    /**
     * @brief Returns group
     * @return Group
     */
    size_t getGroup() const;
    /**
     * @brief Sets group
     * @param group Group
     * @return reference to layer builder
     */
    ConvolutionLayer& setGroup(size_t group);
    /**
     * @brief Return output depth
     * @return Output depth
     */
    size_t getOutDepth() const;
    /**
     * @brief Sets output depth
     * @param outDepth Output depth
     * @return reference to layer builder
     */
    ConvolutionLayer& setOutDepth(size_t outDepth);
};

}  // namespace Builder
}  // namespace InferenceEngine
