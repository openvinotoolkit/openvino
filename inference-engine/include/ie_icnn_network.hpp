// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the ICNNNetwork class
 * @file ie_icnn_network.hpp
 */
#pragma once

#include "ie_common.h"
#include "ie_layers.h"
#include "ie_data.h"
#include "ie_device.hpp"
#include "ie_blob.h"
#include "details/ie_irelease.hpp"
#include "ie_preprocess.hpp"
#include "ie_input_info.hpp"
#include "ie_icnn_network_stats.hpp"
#include "ie_iextension.h"
#include <memory>
#include <map>
#include <string>

namespace InferenceEngine {

/**
 * @brief A collection that contains string as key, and Data smart pointer as value
 */
using OutputsDataMap = std::map<std::string, DataPtr>;

/**
 * @brief This is the main interface to describe the NN topology
 */
class ICNNNetwork : public details::IRelease {
public:
    using Ptr = std::shared_ptr<ICNNNetwork>;

    /**
     * @brief Returns the main network operating precision.
     * This may be MIXED if not homogeneous.
     * @return A precision type
     */
    virtual Precision getPrecision() const noexcept = 0;

    /**
     * @brief Gets the network output Data node information. The received info is stored in the given Data node.
     * For single and multiple outputs networks.
     * @param out Reference to the OutputsDataMap object
     */
    virtual void getOutputsInfo(OutputsDataMap& out) const noexcept  = 0;

    /**
     * @brief Gets the network input Data node information. The received info is stored in the given InputsDataMap object.
     * For single and multiple inputs networks.
     * This method must be called to find out input names for using them later during filling of a map
     * of blobs passed later to InferenceEngine::IInferencePlugin::Infer()
     * @param inputs Reference to InputsDataMap object.
     */
    virtual void getInputsInfo(InputsDataMap& inputs) const noexcept  = 0;


    /**
     * @brief Returns information on certain input pointed by inputName
     * @param inputName Name of input layer to get info on
     * @return A smart pointer to the input information
     */
    virtual InputInfo::Ptr getInput(const std::string& inputName) const noexcept = 0;

    /**
     * @brief Gets the network name. The name is stored in the given pName string.
     * @param pName - will receive actual network name, specified in IR file,
     *     pName should point to valid memory address before invoking this function
     * @param len - size in bytes of pName buffer, actual name is trimmed by this size
     */
    virtual void getName(char* pName, size_t len) const noexcept = 0;

    /**
     * @brief Returns the network name.
     * @return Network name
     */
    virtual const std::string& getName() const noexcept = 0;

    /**
    * @brief Returns the number of layers in the network as an integer value
    * @return The number of layers as an integer value
    */
    virtual size_t layerCount() const noexcept = 0;

    /**
    * @brief Returns a smart pointer reference to a Data node given its name.
     * If the Data node is missing, returns reference to a default initialized new empty data pointer with given name.
    * @param dname Name of the Data node
    * @return Data node smart pointer
    */
    virtual DataPtr& getData(const char* dname)  noexcept = 0;

    /**
    * @brief Insert a layer into the network. A user is responsible to connect it to other data elements.
    * @param layer Const reference to a layer smart pointer
    */
    virtual void addLayer(const CNNLayerPtr& layer) noexcept = 0;

    /**
     * @brief Adds output to the layer
     * @param layerName Name of the layer
     * @param outputIndex Index of the output
     * @param resp Response message
     * @return Status code of the operation
     */
    virtual StatusCode
    addOutput(const std::string& layerName, size_t outputIndex = 0, ResponseDesc* resp = nullptr) noexcept = 0;

    /**
     * @brief Gets network layer with the given name
     * @param layerName Given name of the layer
     * @param out Pointer to the found CNNLayer object with the given name
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept = 0;

    /**
     * @deprecated Deprecated since TargetDevice is deprecated. Specify target device in InferenceEngine::Core directly.
     * @brief Sets a desirable device to perform all work on.
     * Some plug-ins might not support some target devices and may abort execution with an appropriate error message.
     * @param device Device to set as a target
     */
    #ifndef _WIN32
    INFERENCE_ENGINE_DEPRECATED
    #endif
    virtual void setTargetDevice(TargetDevice device) noexcept = 0;

    /**
     * @deprecated Deprecated since TargetDevice is deprecated
     * @brief Gets the target device.
     * If setTargetDevice() was not called before, returns eDefault
     * @return A TargetDevice instance
     */
    #ifndef _WIN32
    INFERENCE_ENGINE_DEPRECATED
    #endif
    virtual TargetDevice getTargetDevice() const noexcept = 0;

    /**
     * @deprecated Use ICNNNetwork::setBatchSize(size_t, ResponseDesc*)
     * @brief Changes the inference batch size
     */
    INFERENCE_ENGINE_DEPRECATED
    virtual StatusCode setBatchSize(const size_t size) noexcept {
        ResponseDesc resp;
        return setBatchSize(size, &resp);
    }

    /**
     * @brief Changes the inference batch size.
     * @note There are several limitations and it's not recommended to use it. Set batch to the input shape and call ICNNNetwork::reshape.
     * @param size Size of batch to set
     * @return Status code of the operation
     * @note: Current implementation of the function sets batch size to the first dimension of all layers in the networks.
     * Before calling it make sure that all your layers have batch in the first dimension, otherwise the method works incorrectly.
     * This limitation is resolved via shape inference feature
     * by using InferenceEngine::ICNNNetwork::reshape method.
     * To read more refer to the Shape Inference section in documentation
     */
    virtual StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept = 0;

    /**
    * @brief Gets the inference batch size
    * @return The size of batch as a size_t value
    */
    virtual size_t getBatchSize() const noexcept = 0;

    /**
     * @brief Map of pairs: name of corresponding data and its dimension.
     */
    using InputShapes = std::map<std::string, SizeVector>;

    /**
     * @brief Run shape inference with new input shapes for the network
     * @param inputShapes - map of pairs: name of corresponding data and its dimension.
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation
     */
    virtual StatusCode reshape(const InputShapes& /*inputShapes*/, ResponseDesc* /*resp*/) noexcept { return NOT_IMPLEMENTED; };

    /**
     * @brief Registers extension within the plugin
     * @param extension Pointer to already loaded reader extension with shape propagation implementations
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode
    AddExtension(const IShapeInferExtensionPtr& /*extension*/, ResponseDesc* /*resp*/) noexcept { return NOT_IMPLEMENTED; };

    virtual StatusCode getStats(ICNNNetworkStats** /*stats*/, ResponseDesc* /*resp*/) const noexcept { return NOT_IMPLEMENTED; };

    /**
     * @brief Serialize network to IR and weights files.
     * @param xmlPath Path to output IR file.
     * @param binPath Path to output weights file.
     * @return Status code of the operation
     */
    virtual StatusCode serialize(const std::string &xmlPath, const std::string &binPath, ResponseDesc* resp) const noexcept = 0;
};
}  // namespace InferenceEngine
