// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_api.h"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"
#include "ie_input_info.hpp"
#include <ie_icnn_network.hpp>
#include <cpp/ie_cnn_network.h>

#include <legacy/ie_layers.h>

namespace InferenceEngine {

namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(CNNNetworkImpl) final : public ICNNNetwork {
public:
    CNNNetworkImpl();
    explicit CNNNetworkImpl(const CNNNetwork & ngraphImpl);
    ~CNNNetworkImpl();

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

    DataPtr& getData(const char* name) noexcept {
        return _data[name];
    }

    void addData(const char* name, DataPtr data) noexcept {
        _data.emplace(name, data);
    }

    DataPtr& getData(const std::string& name) {
        return getData(name.c_str());
    }

    void addLayer(const CNNLayerPtr& layer) noexcept;

    void removeLayer(const std::string& layerName);

    // renames layer, statistics is not supported
    void renameLayer(const std::string& currentName, const std::string& newName);

    void removeData(const std::string& dataName);

    StatusCode getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept;

    // public version
    StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept override;

    // for internal usage (e.g. setBatch via reshape in tests)
    StatusCode setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept;

    size_t getBatchSize() const noexcept override;

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void resolveOutput();

    void addOutput(const std::string& dataName);

    void removeOutput(const std::string& dataName);

    virtual void validate(int = 2);

    StatusCode reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                       ResponseDesc* resp) noexcept override;

    StatusCode serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
        noexcept override;

    StatusCode serialize(std::ostream& xmlBuf, std::ostream& binBuf, ResponseDesc* resp) const
        noexcept override;

    StatusCode serialize(std::ostream& xmlBuf, Blob::Ptr& binBlob, ResponseDesc* resp) const
        noexcept override;

protected:
    std::map<std::string, DataPtr> _data;
    std::map<std::string, CNNLayerPtr> _layers;
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    std::string _name;
    DataPtr _emptyData;
};

IE_SUPPRESS_DEPRECATED_END

typedef std::shared_ptr<CNNNetworkImpl> CNNNetworkImplPtr;

}  // namespace details
}  // namespace InferenceEngine
