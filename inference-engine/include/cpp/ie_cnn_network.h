// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper for ICNNNetwork object
 *
 * @file ie_cnn_network.h
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ie_icnn_network.hpp"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"
#include "details/ie_exception_conversion.hpp"
#include "ie_extension.h"

namespace ngraph {

class Function;

}  // namespace ngraph

namespace InferenceEngine {

/**
 * @brief This class contains all the information about the Neural Network and the related binary information
 */
class INFERENCE_ENGINE_API_CLASS(CNNNetwork) {
public:
    /**
     * @brief A default constructor
     */
    CNNNetwork();

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief Allows helper class to manage lifetime of network object
     *
     * @param network Pointer to the network object
     */
    explicit CNNNetwork(std::shared_ptr<ICNNNetwork> network);
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A constructor from ngraph::Function object
     * This constructor wraps existing ngraph::Function
     * If you want to avoid modification of original Function, please create a copy
     * @param network Pointer to the ngraph::Function object
     * @param exts Vector of pointers to IE extension objects
     */
    explicit CNNNetwork(const std::shared_ptr<ngraph::Function>& network,
                        const std::vector<IExtensionPtr>& exts = {});

    /**
     * @copybrief ICNNNetwork::getOutputsInfo
     *
     * Wraps ICNNNetwork::getOutputsInfo
     *
     * @return outputs Reference to the OutputsDataMap object
     */
    OutputsDataMap getOutputsInfo() const;

    /**
     * @copybrief ICNNNetwork::getInputsInfo
     *
     * Wraps ICNNNetwork::getInputsInfo
     *
     * @return inputs Reference to InputsDataMap object
     */
    InputsDataMap getInputsInfo() const;

    /**
     * @copybrief ICNNNetwork::layerCount
     *
     * Wraps ICNNNetwork::layerCount
     *
     * @return The number of layers as an integer value
     */
    size_t layerCount() const;

    /**
     * @copybrief ICNNNetwork::getName
     *
     * Wraps ICNNNetwork::getName
     *
     * @return Network name
     */
    const std::string& getName() const;

    /**
     * @copybrief ICNNNetwork::setBatchSize
     *
     * Wraps ICNNNetwork::setBatchSize
     *
     * @param size Size of batch to set
     */
    void setBatchSize(const size_t size);

    /**
     * @copybrief ICNNNetwork::getBatchSize
     *
     * Wraps ICNNNetwork::getBatchSize
     *
     * @return The size of batch as a size_t value
     */
    size_t getBatchSize() const;

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @deprecated InferenceEngine::ICNNNetwork interface is deprecated
     * @brief An overloaded operator cast to get pointer on current network
     *
     * @return A shared pointer of the current network
     */
    // INFERENCE_ENGINE_DEPRECATED("InferenceEngine::ICNNNetwork interface is deprecated")
    operator ICNNNetwork::Ptr();

    /**
     * @deprecated InferenceEngine::ICNNNetwork interface is deprecated
     * @brief An overloaded operator & to get current network
     *
     * @return An instance of the current network
     */
    // INFERENCE_ENGINE_DEPRECATED("InferenceEngine::ICNNNetwork interface is deprecated")
    operator ICNNNetwork&();

    /**
     * @deprecated InferenceEngine::ICNNNetwork interface is deprecated
     * @brief An overloaded operator & to get current network
     *
     * @return A const reference of the current network
     */
    // INFERENCE_ENGINE_DEPRECATED("InferenceEngine::ICNNNetwork interface is deprecated")
    operator const ICNNNetwork&() const;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Returns constant nGraph function
     *
     * @return constant nGraph function
     */
    std::shared_ptr<ngraph::Function> getFunction();

    /**
     * @brief Returns constant nGraph function
     *
     * @return constant nGraph function
     */
    std::shared_ptr<const ngraph::Function> getFunction() const;

    /**
     * @copybrief ICNNNetwork::addOutput
     *
     * Wraps ICNNNetwork::addOutput
     *
     * @param layerName Name of the layer
     * @param outputIndex Index of the output
     */
    void addOutput(const std::string& layerName, size_t outputIndex = 0);

    /**
     * @brief Helper method to get collect all input shapes with names of corresponding Data objects
     *
     * @return Map of pairs: input name and its dimension.
     */
    ICNNNetwork::InputShapes getInputShapes() const;

    /**
     * @brief Run shape inference with new input shapes for the network
     *
     * @param inputShapes - map of pairs: name of corresponding data and its dimension.
     */
    void reshape(const ICNNNetwork::InputShapes& inputShapes);

    /**
     * @brief Serialize network to IR and weights files.
     *
     * @param xmlPath Path to output IR file.
     * @param binPath Path to output weights file. The parameter is skipped in case
     * of executable graph info serialization.
     */
    void serialize(const std::string& xmlPath, const std::string& binPath = {}) const;

    /**
     * @brief Method maps framework tensor name to OpenVINO name
     *
     * @param orig_name Framework tensor name
     *
     * @return OpenVINO name
     */
    std::string getOVNameForTensor(const std::string& orig_name) const {
        std::string ov_name;
        CALL_STATUS_FNC(getOVNameForTensor, ov_name, orig_name);
        return ov_name;
    }

    /**
     * @brief Method maps framework operator name to OpenVINO name
     *
     * @param orig_name Framework operation name
     *
     * @return OpenVINO name
     */
    std::string getOVNameForOperation(const std::string& orig_name) const {
        std::string ov_name;
        CALL_STATUS_FNC(getOVNameForOperation, ov_name, orig_name);
        return ov_name;
    }

protected:
    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief Network extra interface, might be nullptr
     */
    std::shared_ptr<ICNNNetwork> network;

    /**
     * @brief A pointer to the current network
     */
    ICNNNetwork* actual = nullptr;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A pointer to output data
     */
    DataPtr output;
};

}  // namespace InferenceEngine
