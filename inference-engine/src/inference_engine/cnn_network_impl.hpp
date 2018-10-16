// Copyright (C) 2018 Intel Corporation
//
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
#include "description_buffer.hpp"
#include <string>
#include <vector>

namespace InferenceEngine {
namespace details {
class INFERENCE_ENGINE_API_CLASS(CNNNetworkImpl) : public ICNNNetwork {
public:
    CNNNetworkImpl();
    Precision getPrecision() const noexcept override {
        return precision;
    }

    void setPrecision(Precision::ePrecision  prec) {
        precision = prec;
    }

    void getOutputsInfo(std::map<std::string, DataPtr> &out) const noexcept override;

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

    size_t layerCount()  const noexcept override {
        return _layers.size();
    }

    DataPtr& getData(const char* name) noexcept override  {
        return _data[name];
    }

    DataPtr& getData(const std::string& name) {
        return getData(name.c_str());
    }

    void addLayer(const CNNLayerPtr& layer) noexcept override;

    StatusCode getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept override;

    // deprecated, as there is no ResponseDesc to put error message
    StatusCode setBatchSize(const size_t size) noexcept override;

    // public version
    StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept override;

    // for internal usage (e.g. setBatch via reshape in tests)
    StatusCode setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept;

    size_t getBatchSize() const noexcept override;

    void setTargetDevice(TargetDevice device) noexcept override {
        _targetDevice = device;
    }

    TargetDevice getTargetDevice() const noexcept override {
        return _targetDevice;
    }

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void resolveOutput();

    void addOutput(const std::string& dataName);

    void Release() noexcept override {
        delete this;
    }

    virtual void validate(int = 2);

    StatusCode reshape(const std::map<std::string, std::vector<size_t>> &inputShapes, ResponseDesc* resp) noexcept override;

    StatusCode
    AddExtension(const InferenceEngine::IShapeInferExtensionPtr &extension, InferenceEngine::ResponseDesc *resp) noexcept override;

protected:
    Precision precision {Precision::MIXED};
    std::map<std::string, DataPtr> _data;
    std::map<std::string, CNNLayerPtr> _layers;
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    std::string _name;
    /// @brief
    TargetDevice _targetDevice;
    DataPtr _emptyData;
    std::vector<IShapeInferExtensionPtr> _shapeInferExts;
};


typedef std::shared_ptr<CNNNetworkImpl> CNNNetworkImplPtr;
}  // namespace details
}  // namespace InferenceEngine
