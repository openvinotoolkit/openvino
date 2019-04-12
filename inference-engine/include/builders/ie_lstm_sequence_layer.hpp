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
 * @brief The class represents a builder for LSTMSequence layer
 */
class INFERENCE_ENGINE_API_CLASS(LSTMSequenceLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit LSTMSequenceLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit LSTMSequenceLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit LSTMSequenceLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    LSTMSequenceLayer& setName(const std::string& name);

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
    LSTMSequenceLayer& setInputPorts(const std::vector<Port>& ports);

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
    LSTMSequenceLayer& setOutputPorts(const std::vector<Port>& ports);

    int getHiddenSize() const;
    LSTMSequenceLayer& setHiddenSize(int size);
    bool getSequenceDim() const;
    LSTMSequenceLayer& setSqquenceDim(bool flag);
    const std::vector<std::string>& getActivations() const;
    LSTMSequenceLayer& setActivations(const std::vector<std::string>& activations);
    const std::vector<float>& getActivationsAlpha() const;
    LSTMSequenceLayer& setActivationsAlpha(const std::vector<float>& activations);
    const std::vector<float>& getActivationsBeta() const;
    LSTMSequenceLayer& setActivationsBeta(const std::vector<float>& activations);
    float getClip() const;
    LSTMSequenceLayer& setClip(float clip);
    bool getInputForget() const;
    LSTMSequenceLayer& setInputForget(bool flag);
    const std::string& getDirection() const;
    LSTMSequenceLayer& setDirection(const std::string& direction);
};

}  // namespace Builder
}  // namespace InferenceEngine


