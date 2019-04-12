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
 * @brief The class represents a builder for RNNSequence layer
 */
class INFERENCE_ENGINE_API_CLASS(RNNSequenceLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit RNNSequenceLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit RNNSequenceLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit RNNSequenceLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    RNNSequenceLayer& setName(const std::string& name);

    /**
     * @brief Returns input ports with shapes for the layer
     * @return Vector of ports
     */
    const std::vector<Port>& getInputPorts() const;
    /**
     * @brief Sets input ports for the layer
     * @param ports vector of input ports
     * @return reference to layer builder
     */
    RNNSequenceLayer& setInputPorts(const std::vector<Port>& ports);

    /**
     * @brief Returns output ports with shapes for the layer
     * @return Vector of ports
     */
    const std::vector<Port>& getOutputPorts() const;
    /**
     * @brief Sets output ports for the layer
     * @param ports vector of output ports
     * @return reference to layer builder
     */
    RNNSequenceLayer& setOutputPorts(const std::vector<Port>& ports);

    int getHiddenSize() const;
    RNNSequenceLayer& setHiddenSize(int size);
    bool getSequenceDim() const;
    RNNSequenceLayer& setSqquenceDim(bool flag);
    const std::vector<std::string>& getActivations() const;
    RNNSequenceLayer& setActivations(const std::vector<std::string>& activations);
    const std::vector<float>& getActivationsAlpha() const;
    RNNSequenceLayer& setActivationsAlpha(const std::vector<float>& activations);
    const std::vector<float>& getActivationsBeta() const;
    RNNSequenceLayer& setActivationsBeta(const std::vector<float>& activations);
    float getClip() const;
    RNNSequenceLayer& setClip(float clip);
};

}  // namespace Builder
}  // namespace InferenceEngine


