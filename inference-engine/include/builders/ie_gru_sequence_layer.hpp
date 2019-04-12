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
 * @brief The class represents a builder for GRUSequence layer
 */
class INFERENCE_ENGINE_API_CLASS(GRUSequenceLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit GRUSequenceLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit GRUSequenceLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit GRUSequenceLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    GRUSequenceLayer& setName(const std::string& name);

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
    GRUSequenceLayer& setInputPorts(const std::vector<Port>& ports);

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
    GRUSequenceLayer& setOutputPorts(const std::vector<Port>& ports);

    int getHiddenSize() const;
    GRUSequenceLayer& setHiddenSize(int size);
    bool getSequenceDim() const;
    GRUSequenceLayer& setSqquenceDim(bool flag);
    const std::vector<std::string>& getActivations() const;
    GRUSequenceLayer& setActivations(const std::vector<std::string>& activations);
    const std::vector<float>& getActivationsAlpha() const;
    GRUSequenceLayer& setActivationsAlpha(const std::vector<float>& activations);
    const std::vector<float>& getActivationsBeta() const;
    GRUSequenceLayer& setActivationsBeta(const std::vector<float>& activations);
    float getClip() const;
    GRUSequenceLayer& setClip(float clip);
    bool getLinearBeforeReset() const;
    GRUSequenceLayer& setLinearBeforeReset(bool flag);
    const std::string& getDirection() const;
    GRUSequenceLayer& setDirection(const std::string& direction);
};

}  // namespace Builder
}  // namespace InferenceEngine


