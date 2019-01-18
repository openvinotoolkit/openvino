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
 * @brief The class represents a builder for CTCGreedyDecoder layer
 */
class INFERENCE_ENGINE_API_CLASS(CTCGreedyDecoderLayer): public LayerFragment {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit CTCGreedyDecoderLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param genLayer generic builder
     */
    explicit CTCGreedyDecoderLayer(Layer& genLayer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    CTCGreedyDecoderLayer& setName(const std::string& name);

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
    CTCGreedyDecoderLayer& setInputPorts(const std::vector<Port>& ports);
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
    CTCGreedyDecoderLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns CTCMergeRepeated
     * @return true if merge repeated
     */
    bool getCTCMergeRepeated() const;
    /**
     * @brief Sets CTCMergeRepeated
     * @param flag bool value
     * @return reference to layer builder
     */
    CTCGreedyDecoderLayer& setCTCMergeRepeated(bool flag);
};

}  // namespace Builder
}  // namespace InferenceEngine

