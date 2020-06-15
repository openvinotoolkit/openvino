// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file containing ngraph implementation of public ICNNNetwork interface
 * @file cnn_network_ngraph_impl.hpp
 */

#pragma once

#include <algorithm>
#include <functional>
#include <ie_icnn_network.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <ngraph/attribute_visitor.hpp>
#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <string>
#include <vector>

#include "cnn_network_impl.hpp"
#include "description_buffer.hpp"
#include "ie_api.h"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"
#include "ie_input_info.hpp"

namespace InferenceEngine {
namespace ShapeInfer {
class Reshaper;

using ReshaperPtr = std::shared_ptr<Reshaper>;
}  // namespace ShapeInfer

namespace details {

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief Ngraph-based implementation of the ICNNNetwork interface.
 */
class INFERENCE_ENGINE_API_CLASS(CNNNetworkNGraphImpl): public ICNNNetwork {
public:
    CNNNetworkNGraphImpl(const std::shared_ptr<::ngraph::Function>& nGraph);
    ~CNNNetworkNGraphImpl() override;

    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Function directly")
    Precision getPrecision() const noexcept override;
    void getOutputsInfo(std::map<std::string, DataPtr>& out) const noexcept override;

    void getInputsInfo(InputsDataMap& inputs) const noexcept override;

    InputInfo::Ptr getInput(const std::string& inputName) const noexcept override;

    INFERENCE_ENGINE_DEPRECATED("Use CNNNetworkNGraphImpl::getName() returning std::string")
    void getName(char* pName, size_t len) const noexcept override;

    const std::string& getName() const noexcept override;

    size_t layerCount() const noexcept override;

    void setInputInfo(InputInfo::Ptr data);

    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Function directly")
    DataPtr& getData(const char* name) noexcept override;

    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Function directly")
    DataPtr& getData(const std::string& name);

    std::shared_ptr<ICNNNetwork> getCNNNetwork();

    // This method is not really implemented; don't call it
    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Function directly")
    void addLayer(const CNNLayerPtr& layer) noexcept override;

    INFERENCE_ENGINE_DEPRECATED("Use ngraph::Function directly")
    StatusCode getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept override;

    // public version
    StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept override;

    // for internal usage (e.g. setBatch via reshape in tests)
    StatusCode setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept;

    size_t getBatchSize() const noexcept override;

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void addOutput(const std::string& dataName);

    StatusCode getStats(ICNNNetworkStats** stats, ResponseDesc* resp) const noexcept override {
        return StatusCode::NOT_FOUND;
    }

    void Release() noexcept override {
        delete this;
    }

    std::shared_ptr<const ::ngraph::Function> getFunction() const noexcept override {
        return !cnnNetwork ? _ngraph_function : nullptr;
    }
    std::shared_ptr<::ngraph::Function> getFunction() noexcept override {
        return !cnnNetwork ? _ngraph_function : nullptr;
    }

    virtual void validate(int = 10);

    StatusCode reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                       ResponseDesc* resp) noexcept override;

    StatusCode AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension,
                            InferenceEngine::ResponseDesc* resp) noexcept override;

    StatusCode serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
        noexcept override;

    void convertToCNNNetworkImpl();

protected:
    std::shared_ptr<::ngraph::Function> _ngraph_function;
    virtual std::shared_ptr<::ngraph::Function> cloneFunction(bool constFolding = false, const std::map<std::string,
            std::vector<size_t>>& inputShapes = {}) const;
    std::mutex& getMutex() {
        return networkMutex;
    }

private:
    std::map<std::string, DataPtr> _data;
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    std::shared_ptr<CNNNetworkImpl> cnnNetwork;
    std::mutex networkMutex;

    /**
     * @brief Create DataPtr for nGraph operation
     *
     * @param output output port from nGraph op
     * @param outName name for DataPtr
     * @param ptr reference to new DataPtr
     */
    void createDataForResult(const ::ngraph::Output<::ngraph::Node>& output, const std::string& outName, DataPtr& ptr);

    friend INFERENCE_ENGINE_API_CPP(std::shared_ptr<CNNNetworkImpl>)
    convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph,
                                 const ICNNNetwork& nGraphImpl, bool keep_constant_inputs);

    /**
     * @brief Reshape on the same shape
     */
    void reshape();
};

class TINGraphBody : public CNNNetworkNGraphImpl {
public:
    explicit TINGraphBody(const std::shared_ptr<::ngraph::Function>& func): CNNNetworkNGraphImpl(func) {}

protected:
    std::shared_ptr<::ngraph::Function> cloneFunction(bool constFolding, const std::map<std::string, std::vector<size_t>>& inputShapes) const override {
        return _ngraph_function;
    }
};

/**
 * @brief Special derived class of Data which converts CNNNetworkNGraphImpl to CNNLayer-based representation
 * in case if a user called Data::getCreatorLayer or Data::getInputTo
 */
class NGraphData : public Data {
public:
    using Ptr = std::shared_ptr<NGraphData>;

    NGraphData(CNNNetworkNGraphImpl* network, const std::string& name, const TensorDesc& desc)
        : Data(name, desc), network(network) {}

    void reset() {
        network = nullptr;
    }

    CNNLayerWeakPtr& getCreatorLayer() override {
        if (Data::getCreatorLayer().lock() == nullptr && network != nullptr) {
            network->convertToCNNNetworkImpl();
        }
        return Data::getCreatorLayer();
    }

    std::map<std::string, CNNLayerPtr>& getInputTo() override {
        if (Data::getInputTo().empty() && network != nullptr) {
            network->convertToCNNNetworkImpl();
        }

        return Data::getInputTo();
    }

private:
    CNNNetworkNGraphImpl* network;
};

IE_SUPPRESS_DEPRECATED_END

typedef std::shared_ptr<CNNNetworkNGraphImpl> CNNNetworkNGraphImplPtr;
}  // namespace details
}  // namespace InferenceEngine
