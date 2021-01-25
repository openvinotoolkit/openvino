// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the ICNNNetwork class
 *
 * @file ie_icnn_network.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"
#include "ie_input_info.hpp"
#include "details/ie_irelease.hpp"

#if defined IMPLEMENT_INFERENCE_ENGINE_API || defined IMPLEMENT_INFERENCE_ENGINE_PLUGIN || 1
# define INFERENCE_ENGINE_ICNNNETWORK_CLASS(...) INFERENCE_ENGINE_API_CLASS(__VA_ARGS__)
#else
# define INFERENCE_ENGINE_ICNNNETWORK_CLASS(...)                                 \
    INFERENCE_ENGINE_INTERNAL("Use InferenceEngine::CNNNetwork wrapper instead") \
    INFERENCE_ENGINE_API_CLASS(__VA_ARGS__)
#endif

namespace ngraph {

class Function;

}  // namespace ngraph

namespace InferenceEngine {

/**
 * @brief A collection that contains string as key, and Data smart pointer as value
 */
using OutputsDataMap = std::map<std::string, DataPtr>;

/**
 * @deprecated Use InferenceEngine::CNNNetwork wrapper instead
 * @interface ICNNNetwork
 * @brief This is the main interface to describe the NN topology
 */
class INFERENCE_ENGINE_ICNNNETWORK_CLASS(ICNNNetwork) : public details::IRelease {
public:
    /**
     * @brief A shared pointer to a ICNNNetwork interface
     */
    using Ptr = std::shared_ptr<ICNNNetwork>;

    /**
     * @brief Returns nGraph function
     * @return nGraph function
     */
    virtual std::shared_ptr<ngraph::Function> getFunction() noexcept = 0;

    /**
     * @brief Returns constant nGraph function
     * @return constant nGraph function
     */
    virtual std::shared_ptr<const ngraph::Function> getFunction() const noexcept = 0;

    /**
     * @brief Gets the network output Data node information. The received info is stored in the given Data node.
     *
     * For single and multiple outputs networks.
     *
     * This method need to be called to find output names for using them later
     * when calling InferenceEngine::InferRequest::GetBlob or InferenceEngine::InferRequest::SetBlob
     *
     *
     * @param out Reference to the OutputsDataMap object
     */
    virtual void getOutputsInfo(OutputsDataMap& out) const noexcept = 0;

    /**
     * @brief Gets the network input Data node information. The received info is stored in the given InputsDataMap
     * object.
     *
     * For single and multiple inputs networks.
     * This method need to be called to find out input names for using them later
     * when calling InferenceEngine::InferRequest::SetBlob
     *
     * @param inputs Reference to InputsDataMap object.
     */
    virtual void getInputsInfo(InputsDataMap& inputs) const noexcept = 0;

    /**
     * @brief Returns information on certain input pointed by inputName
     *
     * @param inputName Name of input layer to get info on
     * @return A smart pointer to the input information
     */
    virtual InputInfo::Ptr getInput(const std::string& inputName) const noexcept = 0;

    /**
     * @brief Returns the network name.
     *
     * @return Network name
     */
    virtual const std::string& getName() const noexcept = 0;

    /**
     * @brief Returns the number of layers in the network as an integer value
     *
     * @return The number of layers as an integer value
     */
    virtual size_t layerCount() const noexcept = 0;

    /**
     * @brief Adds output to the layer
     *
     * @param layerName Name of the layer
     * @param outputIndex Index of the output
     * @param resp Response message
     * @return Status code of the operation
     */
    virtual StatusCode addOutput(const std::string& layerName, size_t outputIndex = 0,
                                 ResponseDesc* resp = nullptr) noexcept = 0;

    /**
     * @brief Changes the inference batch size.
     *
     * @note There are several limitations and it's not recommended to use it. Set batch to the input shape and call
     * ICNNNetwork::reshape.
     *
     * @param size Size of batch to set
     * @param responseDesc Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation
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
    virtual StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept = 0;

    /**
     * @brief Gets the inference batch size
     *
     * @return The size of batch as a size_t value
     */
    virtual size_t getBatchSize() const noexcept = 0;

    /**
     * @brief Map of pairs: name of corresponding data and its dimension.
     */
    using InputShapes = std::map<std::string, SizeVector>;

    /**
     * @brief Run shape inference with new input shapes for the network
     *
     * @param inputShapes - map of pairs: name of corresponding data and its dimension.
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation
     */
    virtual StatusCode reshape(const InputShapes& inputShapes, ResponseDesc* resp) noexcept {
        (void)inputShapes;
        (void)resp;
        return NOT_IMPLEMENTED;
    };

    /**
     * @brief Serialize network to IR and weights files.
     *
     * @param xmlPath Path to output IR file.
     * @param binPath Path to output weights file.
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation
     */
    virtual StatusCode serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
        noexcept = 0;

    /**
     * @brief A virtual destructor.
     */
    virtual ~ICNNNetwork();
};
}  // namespace InferenceEngine
