// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <ie_icnn_network.hpp>
#include "ie_common.h"
#include "ie_data.h"
#include "ie_blob.h"
#include "ie_api.h"
#include "ie_input_info.hpp"
#include "description_buffer.hpp"
#include <string>
#include <vector>

#include "cnn_network_stats_impl.hpp"
#include "cnn_network_impl.hpp"

namespace ngraph {
class Function;
}  // namespace ngraph

namespace InferenceEngine {
namespace ShapeInfer {
class Reshaper;

using ReshaperPtr = std::shared_ptr<Reshaper>;
}  // namespace ShapeInfer
namespace details {
class INFERENCE_ENGINE_API_CLASS(CNNNetworkNGraphImpl) : public ICNNNetwork {
public:
    CNNNetworkNGraphImpl(const std::shared_ptr<ngraph::Function>& nGraph);
    ~CNNNetworkNGraphImpl() override;

    Precision getPrecision() const noexcept override;
    void getOutputsInfo(std::map<std::string, DataPtr> &out) const noexcept override;

    void getInputsInfo(InputsDataMap& inputs) const noexcept override;

    InputInfo::Ptr getInput(const std::string& inputName) const noexcept override;

    void setInputInfo(InputInfo::Ptr data) {
        if (cnnNetwork)
            cnnNetwork->setInputInfo(data);
        _inputData[data->name()] = data;
    }

    void getName(char* pName, size_t len) const noexcept override;

    const std::string& getName() const noexcept override;

    size_t layerCount()  const noexcept override;

    DataPtr& getData(const char* name) noexcept override  {
        if (cnnNetwork)
            return cnnNetwork->getData(name);
        return _data[name];
    }

    DataPtr& getData(const std::string& name) {
        return getData(name.c_str());
    }

    // This method is not really implemented; don't call it
    void addLayer(const CNNLayerPtr& layer) noexcept override;

    StatusCode getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept override;

    // public version
    StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept override;

    // for internal usage (e.g. setBatch via reshape in tests)
    StatusCode setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept;

    size_t getBatchSize() const noexcept override;

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void addOutput(const std::string& dataName);
    StatusCode getStats(ICNNNetworkStats** stats, ResponseDesc* resp) const noexcept override {
        if (cnnNetwork) {
            return cnnNetwork->getStats(stats, resp);
        }
        if (stats == nullptr) return StatusCode::PARAMETER_MISMATCH;
        *stats = _stats.get();
        return StatusCode::OK;
    }

    void Release() noexcept override {
        delete this;
    }

    const std::shared_ptr<ngraph::Function> getFunction() const noexcept override {
        if (cnnNetwork)
            return nullptr;
        return _ngraph_function;
    }

    std::shared_ptr<ngraph::Function> getFunction() noexcept override {
        if (cnnNetwork)
            return nullptr;
        return _ngraph_function;
    }


    virtual void validate(int = 10);

    StatusCode reshape(const std::map<std::string, std::vector<size_t>> &inputShapes, ResponseDesc* resp) noexcept override;

    StatusCode
    AddExtension(const InferenceEngine::IShapeInferExtensionPtr &extension, InferenceEngine::ResponseDesc *resp) noexcept override;

    StatusCode serialize(const std::string &xmlPath, const std::string &binPath, ResponseDesc* resp) const noexcept override;

    std::shared_ptr<CNNNetworkImpl> convertToCNNNetworkImpl() const;
    std::shared_ptr<CNNNetworkImpl> convertToCNNNetworkImpl();

protected:
    std::map<std::string, DataPtr> _data;
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    DataPtr _emptyData;
    CNNNetworkStatsImplPtr _stats;
    std::shared_ptr<CNNNetworkImpl> cnnNetwork;
    std::shared_ptr<ngraph::Function> _ngraph_function;
};


typedef std::shared_ptr<CNNNetworkNGraphImpl> CNNNetworkNGraphImplPtr;
}  // namespace details
}  // namespace InferenceEngine
