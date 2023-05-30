// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper for ICNNNetwork object
 *
 * @file ie_cnn_network.h
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"
#include "ie_icnn_network.hpp"
#include "ngraph/function.hpp"

namespace InferenceEngine {

class IExtension;

/**
 * @brief This class contains all the information about the Neural Network and the related binary information
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(CNNNetwork) {
public:
    /**
     * @brief A default constructor
     */
    CNNNetwork();

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @deprecated Don't use this constructor. It will be removed soon
     * @brief Allows helper class to manage lifetime of network object
     *
     * @param network Pointer to the network object
     */
    INFERENCE_ENGINE_DEPRECATED("Don't use this constructor. It will be removed soon")
    explicit CNNNetwork(std::shared_ptr<ICNNNetwork> network);

    /**
     * @brief A constructor from ngraph::Function object
     * This constructor wraps existing ngraph::Function
     * If you want to avoid modification of original Function, please create a copy
     * @param network Pointer to the ngraph::Function object
     * @param exts Vector of pointers to IE extension objects
     */
    explicit CNNNetwork(const std::shared_ptr<ngraph::Function>& network,
                        const std::vector<std::shared_ptr<IExtension>>& exts = {});
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Gets the network output Data node information. The received info is stored in the given Data node.
     *
     * For single and multiple outputs networks.
     *
     * This method need to be called to find out OpenVINO output names for using them later
     * when calling InferenceEngine::InferRequest::GetBlob or InferenceEngine::InferRequest::SetBlob
     *
     * If you want to use framework names, you can use InferenceEngine::CNNNetwork::getOVNameForTensor
     * method to map framework names to OpenVINO names
     *
     * @return the InferenceEngine::OutputsDataMap object
     */
    OutputsDataMap getOutputsInfo() const;

    /**
     * @brief Gets the network input Data node information. The received info is stored in the given InputsDataMap
     * object.
     *
     * For single and multiple inputs networks.
     * This method need to be called to find out OpenVINO input names for using them later
     * when calling InferenceEngine::InferRequest::SetBlob
     *
     * If you want to use framework names, you can use InferenceEngine::ICNNNetwork::getOVNameForTensor
     * method to map framework names to OpenVINO names
     *
     * @return The InferenceEngine::InputsDataMap object.
     */
    InputsDataMap getInputsInfo() const;

    /**
     * @brief Returns the number of layers in the network as an integer value
     * @return The number of layers as an integer value
     */
    size_t layerCount() const;

    /**
     * @brief Returns the network name.
     * @return Network name
     */
    const std::string& getName() const;

    /**
     * @brief Changes the inference batch size.
     *
     * @note There are several limitations and it's not recommended to use it. Set batch to the input shape and call
     * InferenceEngine::CNNNetwork::reshape.
     *
     * @param size Size of batch to set
     *
     * @note Current implementation of the function sets batch size to the first dimension of all layers in the
     * networks. Before calling it make sure that all your layers have batch in the first dimension, otherwise the
     * method works incorrectly. This limitation is resolved via shape inference feature by using
     * InferenceEngine::ICNNNetwork::reshape method. To read more refer to the Shape Inference section in documentation
     *
     * @note Current implementation of the function sets batch size to the first dimension of all layers in the
     * networks. Before calling it make sure that all your layers have batch in the first dimension, otherwise the
     * method works incorrectly. This limitation is resolved via shape inference feature by using
     * InferenceEngine::ICNNNetwork::reshape method. To read more refer to the Shape Inference section in documentation
     */
    void setBatchSize(const size_t size);

    /**
     * @brief Gets the inference batch size
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
    INFERENCE_ENGINE_DEPRECATED("InferenceEngine::ICNNNetwork interface is deprecated")
    operator ICNNNetwork::Ptr();

    /**
     * @deprecated InferenceEngine::ICNNNetwork interface is deprecated
     * @brief An overloaded operator & to get current network
     *
     * @return An instance of the current network
     */
    INFERENCE_ENGINE_DEPRECATED("InferenceEngine::ICNNNetwork interface is deprecated")
    operator ICNNNetwork&();

    /**
     * @deprecated InferenceEngine::ICNNNetwork interface is deprecated
     * @brief An overloaded operator & to get current network
     *
     * @return A const reference of the current network
     */
    INFERENCE_ENGINE_DEPRECATED("InferenceEngine::ICNNNetwork interface is deprecated")
    operator const ICNNNetwork&() const;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Returns constant nGraph function
     * @return constant nGraph function
     */
    std::shared_ptr<ngraph::Function> getFunction();

    /**
     * @brief Returns constant nGraph function
     * @return constant nGraph function
     */
    std::shared_ptr<const ngraph::Function> getFunction() const;

    /**
     * @brief Adds output to the layer
     * @param layerName Name of the layer
     * @param outputIndex Index of the output
     */
    void addOutput(const std::string& layerName, size_t outputIndex = 0);

    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief Helper method to get collect all input shapes with names of corresponding Data objects
     * @return Map of pairs: input name and its dimension.
     */
    ICNNNetwork::InputShapes getInputShapes() const;

    /**
     * @brief Run shape inference with new input shapes for the network
     * @param inputShapes A map of pairs: name of corresponding data and its dimension.
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
     * @brief Serialize network to IR and weights streams.
     *
     * @param xmlBuf output IR stream.
     * @param binBuf output weights stream.
     */
    void serialize(std::ostream& xmlBuf, std::ostream& binBuf) const;

    /**
     * @brief Serialize network to IR stream and weights Blob::Ptr.
     *
     * @param xmlBuf output IR stream.
     * @param binBlob output weights Blob::Ptr.
     */
    void serialize(std::ostream& xmlBuf, Blob::Ptr& binBlob) const;

    /**
     * @brief Method maps framework tensor name to OpenVINO name
     * @param orig_name Framework tensor name
     * @return OpenVINO name
     */
    std::string getOVNameForTensor(const std::string& orig_name) const;

private:
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
};

}  // namespace InferenceEngine
