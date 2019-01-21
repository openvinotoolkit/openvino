// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_fragment.hpp>
#include <ie_inetwork.hpp>
#include <vector>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for ArgMax layer
 */
class INFERENCE_ENGINE_API_CLASS(ConvolutionLayer): public LayerFragment {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ConvolutionLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param genLayer generic builder
     */
    explicit ConvolutionLayer(Layer& genLayer);
    /**
     * @brief Operator creates generic layer builder
     * @return Generic layer builder
     */
    operator Layer() const override;
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ConvolutionLayer& setName(const std::string& name);

    /**
     * @brief Sets weights for layer
     * @param weights Constant blob with weights
     * @return reference to layer builder
     */
    ConvolutionLayer& setWeights(const Blob::CPtr& weights);
    /**
     * @brief Sets biases for layer
     * @param biases Constant blob with biases
     * @return reference to layer builder
     */
    ConvolutionLayer& setBiases(const Blob::CPtr& biases);

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

    /**
     * @brief Validates layer before creation
     * @param layer generic layer builder
     */
    static void validate(const Layer& layer);
};

}  // namespace Builder
}  // namespace InferenceEngine
