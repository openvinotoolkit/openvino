// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper for ICNNNetwork object
 *
 * @file ie_cnn_network.h
 */
#pragma once

#include <ie_icnn_net_reader.h>

#include <details/ie_cnn_network_iterator.hpp>
#include <details/ie_exception_conversion.hpp>
#include <ie_icnn_network.hpp>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"

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
    CNNNetwork() = default;

    /**
     * @brief Allows helper class to manage lifetime of network object
     *
     * @param network Pointer to the network object
     */
    explicit CNNNetwork(std::shared_ptr<ICNNNetwork> network): network(network) {
        actual = network.get();
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
        }
    }

    /**
     * @brief A constructor from ngraph::Function object
     * @param network Pointer to the ngraph::Function object
     */
    explicit CNNNetwork(const std::shared_ptr<const ngraph::Function>& network);

    /**
     * @brief A constructor from ICNNNetReader object
     *
     * @param reader Pointer to the ICNNNetReader object
     */
    IE_SUPPRESS_DEPRECATED_START
    explicit CNNNetwork(std::shared_ptr<ICNNNetReader> reader): reader(reader), actual(reader->getNetwork(nullptr)) {
        if (actual == nullptr) {
            THROW_IE_EXCEPTION << "CNNNetwork was not initialized.";
        }
    }
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A destructor
     */
    virtual ~CNNNetwork() {}

    /**
     * @deprecated Network precision does not make sence, use precision on egdes. The method will be removed in 2020.3
     * @copybrief ICNNNetwork::getPrecision
     *
     * Wraps ICNNNetwork::getPrecision
     *
     * @return A precision type
     */
    INFERENCE_ENGINE_DEPRECATED("Network precision does not make sence, use precision on egdes. The method will be removed in 2020.3")
    virtual Precision getPrecision() const;

    /**
     * @copybrief ICNNNetwork::getOutputsInfo
     *
     * Wraps ICNNNetwork::getOutputsInfo
     *
     * @return outputs Reference to the OutputsDataMap object
     */
    virtual OutputsDataMap getOutputsInfo() const {
        OutputsDataMap outputs;
        actual->getOutputsInfo(outputs);
        return outputs;
    }

    /**
     * @copybrief ICNNNetwork::getInputsInfo
     *
     * Wraps ICNNNetwork::getInputsInfo
     *
     * @return inputs Reference to InputsDataMap object
     */
    virtual InputsDataMap getInputsInfo() const {
        InputsDataMap inputs;
        actual->getInputsInfo(inputs);
        return inputs;
    }

    /**
     * @copybrief ICNNNetwork::layerCount
     *
     * Wraps ICNNNetwork::layerCount
     *
     * @return The number of layers as an integer value
     */
    size_t layerCount() const {
        return actual->layerCount();
    }

    /**
     * @copybrief ICNNNetwork::getName
     *
     * Wraps ICNNNetwork::getName
     *
     * @return Network name
     */
    const std::string& getName() const noexcept {
        return actual->getName();
    }

    /**
     * @copybrief ICNNNetwork::setBatchSize
     *
     * Wraps ICNNNetwork::setBatchSize
     *
     * @param size Size of batch to set
     * @return Status code of the operation
     */
    virtual void setBatchSize(const size_t size) {
        CALL_STATUS_FNC(setBatchSize, size);
    }

    /**
     * @copybrief ICNNNetwork::getBatchSize
     *
     * Wraps ICNNNetwork::getBatchSize
     *
     * @return The size of batch as a size_t value
     */
    virtual size_t getBatchSize() const {
        return actual->getBatchSize();
    }

    /**
     * @brief An overloaded operator & to get current network
     *
     * @return An instance of the current network
     */
    operator ICNNNetwork&() {
        return *actual;
    }

    /**
     * @brief An overloaded operator & to get current network
     *
     * @return A const reference of the current network
     */
    operator const ICNNNetwork&() const {
        return *actual;
    }

    /**
     * @brief Returns constant nGraph function
     *
     * @return constant nGraph function
     */
    std::shared_ptr<const ngraph::Function> getFunction() const noexcept {
        return actual->getFunction();
    }

    /**
     * @copybrief ICNNNetwork::addOutput
     *
     * Wraps ICNNNetwork::addOutput
     *
     * @param layerName Name of the layer
     * @param outputIndex Index of the output
     */
    void addOutput(const std::string& layerName, size_t outputIndex = 0) {
        CALL_STATUS_FNC(addOutput, layerName, outputIndex);
    }

    /**
     * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2020.3
     * @copybrief ICNNNetwork::getLayerByName
     *
     * Wraps ICNNNetwork::getLayerByName
     *
     * @param layerName Given name of the layer
     * @return Status code of the operation. InferenceEngine::OK if succeeded
     */
    INFERENCE_ENGINE_DEPRECATED("Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2020.3")
    CNNLayerPtr getLayerByName(const char* layerName) const;

    /**
     * @deprecated Use CNNNetwork::getFunction() and work with ngraph::Function directly. The method will be removed in 2020.3
     * @brief Begin layer iterator
     *
     * Order of layers is implementation specific,
     * and can be changed in future
     *
     * @return Iterator pointing to a layer
     */
    IE_SUPPRESS_DEPRECATED_START
    INFERENCE_ENGINE_DEPRECATED("Use CNNNetwork::getFunction() and work with ngraph::Function directly. The method will be removed in 2020.3")
    details::CNNNetworkIterator begin() const;

    /**
     * @deprecated Use CNNNetwork::getFunction() and work with ngraph::Function directly. The method will be removed in 2020.3
     * @brief End layer iterator
     * @return Iterator pointing to a layer
     */
    INFERENCE_ENGINE_DEPRECATED("Use CNNNetwork::getFunction() and work with ngraph::Function directly. The method will be removed in 2020.3")
    details::CNNNetworkIterator end() const;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @deprecated Use CNNNetwork::layerCount() instead. The method will be removed in 2020.3
     * @brief Number of layers in network object
     *
     * @return Number of layers.
     */
    INFERENCE_ENGINE_DEPRECATED("Use CNNNetwork::layerCount() instead. The method will be removed in 2020.3")
    size_t size() const;

    /**
     * @deprecated Use Core::AddExtension to add an extension to the library
     * @brief Registers extension within the plugin
     *
     * @param extension Pointer to already loaded reader extension with shape propagation implementations
     */
    INFERENCE_ENGINE_DEPRECATED("Use Core::AddExtension to add an extension to the library")
    void AddExtension(InferenceEngine::IShapeInferExtensionPtr extension);

    /**
     * @brief Helper method to get collect all input shapes with names of corresponding Data objects
     *
     * @return Map of pairs: input name and its dimension.
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
     *
     * @param inputShapes - map of pairs: name of corresponding data and its dimension.
     */
    virtual void reshape(const ICNNNetwork::InputShapes& inputShapes) {
        CALL_STATUS_FNC(reshape, inputShapes);
    }

    /**
     * @brief Serialize network to IR and weights files.
     *
     * @param xmlPath Path to output IR file.
     * @param binPath Path to output weights file. The parameter is skipped in case
     * of executable graph info serialization.
     */
    void serialize(const std::string& xmlPath, const std::string& binPath = "") const {
        CALL_STATUS_FNC(serialize, xmlPath, binPath);
    }

protected:
    /**
     * @brief Reader extra reference, might be nullptr
     */
    IE_SUPPRESS_DEPRECATED_START
    std::shared_ptr<ICNNNetReader> reader;
    IE_SUPPRESS_DEPRECATED_END
    /**
     * @brief Network extra interface, might be nullptr
     */
    std::shared_ptr<ICNNNetwork> network;

    /**
     * @brief A pointer to the current network
     */
    ICNNNetwork* actual = nullptr;
    /**
     * @brief A pointer to output data
     */
    DataPtr output;
};

}  // namespace InferenceEngine
