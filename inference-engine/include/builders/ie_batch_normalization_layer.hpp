// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_fragment.hpp>
#include <ie_inetwork.hpp>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for BatchNormalization layer
 */
class INFERENCE_ENGINE_API_CLASS(BatchNormalizationLayer): public LayerFragment {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit BatchNormalizationLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param genLayer generic builder
     */
    explicit BatchNormalizationLayer(Layer& genLayer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    BatchNormalizationLayer& setName(const std::string& name);

    /**
     * @brief Returns port with shapes for the layer
     * @return Port with shapes
     */
    const Port& getPort() const;
    /**
     * @brief Sets port shapes for the layer
     * @param port Port with shapes
     * @return reference to layer builder
     */
    BatchNormalizationLayer& setPort(const Port &port);

    /**
     * @brief Sets weights for layer
     * @param weights Constant blob with weights
     * @return reference to layer builder
     */
    BatchNormalizationLayer& setWeights(const Blob::CPtr& weights);
    /**
     * @brief Sets biases for layer
     * @param biases Constant blob with biases
     * @return reference to layer builder
     */
    BatchNormalizationLayer& setBiases(const Blob::CPtr& biases);

    /**
     * @brief Returns epsilon
     * @return Epsilon
     */
    float getEpsilon() const;
    /**
     * @brief Sets epsilon
     * @param eps Epsilon
     * @return reference to layer builder
     */
    BatchNormalizationLayer& setEpsilon(float eps);

    /**
     * @brief Validates layer before creation
     * @param layer generic layer builder
     */
    static void validate(const Layer& layer);
};

}  // namespace Builder
}  // namespace InferenceEngine
