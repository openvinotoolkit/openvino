// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper for ICNNNetwork object
 * @file ie_cnn_network.h
 */
#pragma once

#include <details/ie_exception_conversion.hpp>
#include <details/ie_cnn_network_iterator.hpp>
#include <ie_icnn_network.hpp>
#include <ie_icnn_net_reader.h>
#include "ie_common.h"
#include "ie_data.h"
#include "ie_blob.h"
#include <vector>
#include <string>
#include <map>
#include <utility>
#include <memory>

namespace InferenceEngine {

/**
 * @brief This class contains all the information about the Neural Network and the related binary information
 */
class CNNNetwork {
public:
    /**
     * @brief A default constructor
     */
    CNNNetwork() = default;

    /**
     * @deprecated Use CNNNetwork::CNNNetwork(std::shared_ptr<ICNNNetwork>) to construct a network
     * @brief Initialises helper class from externally managed pointer
     * @param actual Pointer to the network object
     */
    INFERENCE_ENGINE_DEPRECATED
    explicit CNNNetwork(ICNNNetwork* actual) : actual(actual) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
        }
    }

    /**
     * @brief Allows helper class to manage lifetime of network object
     * @param network Pointer to the network object
     */
    explicit CNNNetwork(std::shared_ptr<ICNNNetwork> network)
        : network(network) {
        actual = network.get();
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
        }
    }

    /**
     * @brief A constructor from ICNNNetReader object
     * @param reader Pointer to the ICNNNetReader object
     */
    explicit CNNNetwork(std::shared_ptr<ICNNNetReader> reader)
            : reader(reader)
            , actual(reader->getNetwork(nullptr)) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
        }
    }

    /**
     * @brief A destructor
     */
    virtual ~CNNNetwork() {}

    /**
     * @brief Wraps original method
     * ICNNNetwork::getPrecision
     */
    virtual Precision getPrecision() const {
        return actual->getPrecision();
    }

    /**
     * @brief Wraps original method
     * ICNNNetwork::getOutputsInfo
     */
    virtual OutputsDataMap getOutputsInfo() const {
        OutputsDataMap outputs;
        actual->getOutputsInfo(outputs);
        return outputs;
    }

    /**
     * @brief Wraps original method
     * ICNNNetwork::getInputsInfo
     */
    virtual InputsDataMap getInputsInfo() const {
        InputsDataMap inputs;
        actual->getInputsInfo(inputs);
        return inputs;
    }

    /**
     * @brief Wraps original method
     * ICNNNetwork::layerCount
     */
    size_t layerCount() const {
        return actual->layerCount();
    }

    /**
     * @brief Wraps original method
     * ICNNNetwork::getName
     */
    const std::string& getName() const noexcept {
        return actual->getName();
    }

    /**
     * @brief Wraps original method
     * ICNNNetwork::setBatchSize
     */
    virtual void setBatchSize(const size_t size) {
        CALL_STATUS_FNC(setBatchSize, size);
    }

    /**
     * @brief Wraps original method
     * ICNNNetwork::getBatchSize
     */
    virtual size_t getBatchSize() const {
        return actual->getBatchSize();
    }

    /**
     * @brief An overloaded operator & to get current network
     * @return An instance of the current network
     */
    operator ICNNNetwork &() const {
        return *actual;
    }

    /**
     * @deprecated No needs to specify target device to the network. Use InferenceEngine::Core with target device directly
     * @brief Sets tha target device
     * @param device Device instance to set
     */
    #ifndef _WIN32
    INFERENCE_ENGINE_DEPRECATED
    #endif
    void setTargetDevice(TargetDevice device) {
        IE_SUPPRESS_DEPRECATED_START
        actual->setTargetDevice(device);
        IE_SUPPRESS_DEPRECATED_END
    }

    /**
     * @brief Wraps original method
     * ICNNNetwork::addOutput
     */
    void addOutput(const std::string &layerName, size_t outputIndex = 0) {
        CALL_STATUS_FNC(addOutput, layerName, outputIndex);
    }

    /**
     * @brief Wraps original method
     * ICNNNetwork::getLayerByName
     */
    CNNLayerPtr getLayerByName(const char *layerName) const {
        CNNLayerPtr layer;
        CALL_STATUS_FNC(getLayerByName, layerName, layer);
        return layer;
    }

    /**
     * @brief Begin layer iterator
     * Order of layers is implementation specific,
     * and can be changed in future
     */
    details::CNNNetworkIterator begin() const {
        return details::CNNNetworkIterator(actual);
    }

    /**
     * @brief End layer iterator
     */
    details::CNNNetworkIterator end() const {
        return details::CNNNetworkIterator();
    }

    /**
     * @brief number of layers in network object
     * @return
     */
    size_t size() const {
        return std::distance(std::begin(*this), std::end(*this));
    }

    /**
     * @brief Registers extension within the plugin
     * @param extension Pointer to already loaded reader extension with shape propagation implementations
     */
    void AddExtension(InferenceEngine::IShapeInferExtensionPtr extension) {
        CALL_STATUS_FNC(AddExtension, extension);
    }

    /**
     * @brief - Helper method to get collect all input shapes with names of corresponding Data objects
     * @return Map of pairs: input's name and its dimension.
     */
    virtual ICNNNetwork::InputShapes getInputShapes() const {
        ICNNNetwork::InputShapes shapes;
        InputsDataMap inputs;
        actual->getInputsInfo(inputs);
        for (const auto& pair : inputs) {
            auto info = pair.second;
            if (info) {
                auto data = info->getInputData();
                if (data) {
                    shapes[data->getName()] = data->getTensorDesc().getDims();
                }
            }
        }
        return shapes;
    }

    /**
     * @brief Run shape inference with new input shapes for the network
     * @param inputShapes - map of pairs: name of corresponding data and its dimension.
     */
    virtual void reshape(const ICNNNetwork::InputShapes &inputShapes) {
        CALL_STATUS_FNC(reshape, inputShapes);
    }

    /**
     * @brief Serialize network to IR and weights files.
     * @param xmlPath Path to output IR file.
     * @param binPath Path to output weights file. The parameter is skipped in case
     * of executable graph info serialization.
     */
    void serialize(const std::string &xmlPath, const std::string &binPath = "") const {
        CALL_STATUS_FNC(serialize, xmlPath, binPath);
    }

protected:
    /**
     * @brief reader extra reference, might be nullptr
     */
    std::shared_ptr<ICNNNetReader> reader;
    /**
     * @brief network extra interface, might be nullptr
     */
    std::shared_ptr<ICNNNetwork> network;

    /**
     * @brief A pointer to the current network
     */
    ICNNNetwork *actual = nullptr;
    /**
     * @brief A pointer to output data
     */
    DataPtr output;
};

}  // namespace InferenceEngine
