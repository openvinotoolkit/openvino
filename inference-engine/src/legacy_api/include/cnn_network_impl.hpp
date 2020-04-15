// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_icnn_network.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cnn_network_stats_impl.hpp"
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

class INFERENCE_ENGINE_API_CLASS(CNNNetworkImpl): public ICNNNetwork {
public:
    CNNNetworkImpl();
    ~CNNNetworkImpl() override;
    Precision getPrecision() const noexcept override {
        return precision;
    }

    void setPrecision(Precision::ePrecision prec) {
        precision = prec;
    }

    std::shared_ptr<::ngraph::Function> getFunction() noexcept override {
        return nullptr;
    }

    std::shared_ptr<const ::ngraph::Function> getFunction() const noexcept override {
        return nullptr;
    }

    void getOutputsInfo(std::map<std::string, DataPtr>& out) const noexcept override;

    void getInputsInfo(InputsDataMap& inputs) const noexcept override;

    InputInfo::Ptr getInput(const std::string& inputName) const noexcept override {
        auto it = _inputData.find(inputName);
        if (it == _inputData.end()) {
            return nullptr;
        }
        return it->second;
    }

    void setInputInfo(InputInfo::Ptr data) {
        _inputData[data->name()] = data;
    }

    void removeInputInfo(const std::string& name) {
        _inputData.erase(name);
    }

    void getName(char* pName, size_t len) const noexcept override {
        // Description buffer will preserve garbage if external pointer not initialized
        if (len < 1) return;
        memset(pName, 0, len);
        DescriptionBuffer(pName, len) << _name;
    }

    const std::string& getName() const noexcept override {
        return _name;
    }

    void setName(const std::string& name) {
        _name = name;
    }

    const std::map<std::string, CNNLayerPtr>& allLayers() const {
        return _layers;
    }

    size_t layerCount() const noexcept override {
        return _layers.size();
    }

    DataPtr& getData(const char* name) noexcept override {
        return _data[name];
    }

    void addData(const char* name, DataPtr data) noexcept {
        _data.emplace(name, data);
    }

    DataPtr& getData(const std::string& name) {
        return getData(name.c_str());
    }

    void addLayer(const CNNLayerPtr& layer) noexcept override;

    void removeLayer(const std::string& layerName);

    // renames layer, statistics is not supported
    void renameLayer(const std::string& currentName, const std::string& newName);

    void removeData(const std::string& dataName);

    StatusCode getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept override;

    // public version
    StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept override;

    // for internal usage (e.g. setBatch via reshape in tests)
    StatusCode setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept;

    size_t getBatchSize() const noexcept override;

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void resolveOutput();

    void addOutput(const std::string& dataName);

    void removeOutput(const std::string& dataName);

    StatusCode getStats(ICNNNetworkStats** stats, ResponseDesc* /* resp */) const noexcept override {
        if (stats == nullptr) return StatusCode::PARAMETER_MISMATCH;
        *stats = _stats.get();
        return StatusCode::OK;
    }

    void Release() noexcept override {
        delete this;
    }

    virtual void validate(int = 2);

    StatusCode reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                       ResponseDesc* resp) noexcept override;

    StatusCode AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension,
                            InferenceEngine::ResponseDesc* resp) noexcept override;

    StatusCode serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
        noexcept override;

protected:
    Precision precision {Precision::MIXED};
    std::map<std::string, DataPtr> _data;
    std::map<std::string, CNNLayerPtr> _layers;
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    std::string _name;
    DataPtr _emptyData;
    ShapeInfer::ReshaperPtr _reshaper;
    CNNNetworkStatsImplPtr _stats;
};

IE_SUPPRESS_DEPRECATED_END

typedef std::shared_ptr<CNNNetworkImpl> CNNNetworkImplPtr;
}  // namespace details
}  // namespace InferenceEngine
