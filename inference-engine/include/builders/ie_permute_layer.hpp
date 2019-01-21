// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_fragment.hpp>
#include <vector>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for Permute layer
 */
class INFERENCE_ENGINE_API_CLASS(PermuteLayer): public LayerFragment {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit PermuteLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param genLayer generic builder
     */
    explicit PermuteLayer(Layer& genLayer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    PermuteLayer& setName(const std::string& name);

    /**
     * @brief Sets weights for layer
     * @param weights Constant blob with weights
     * @return reference to layer builder
     */
    PermuteLayer& setWeights(const Blob::CPtr& weights);
    /**
     * @brief Sets biases for layer
     * @param biases Constant blob with biases
     * @return reference to layer builder
     */
    PermuteLayer& setBiases(const Blob::CPtr& biases);

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
    PermuteLayer& setInputPort(const Port& port);
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
    PermuteLayer& setOutputPort(const Port& port);
    /**
     * @brief Return vector of dimensions indexes for output blob
     * @return Order of dimensions for output blob
     */
    const std::vector<size_t> getOrder() const;
    /**
     * @brief Sets the order of dimensions for output blob
     * @param order dimensions indexes for output blob
     * @return reference to layer builder
     */
    PermuteLayer& setOrder(const std::vector<size_t>& order);
};

}  // namespace Builder
}  // namespace InferenceEngine

