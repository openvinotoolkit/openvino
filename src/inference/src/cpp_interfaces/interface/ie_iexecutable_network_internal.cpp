// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp/ie_cnn_network.h"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_parameter.hpp"
#include "openvino/core/node.hpp"

namespace InferenceEngine {

void IExecutableNetworkInternal::setNetworkInputs(const InputsDataMap& networkInputs) {
    _networkInputs = networkInputs;
}

void IExecutableNetworkInternal::setNetworkOutputs(const OutputsDataMap& networkOutputs) {
    _networkOutputs = networkOutputs;
}

void IExecutableNetworkInternal::setInputs(const std::vector<std::shared_ptr<const ov::Node>>& params) {
    _parameters = params;
}
const std::vector<std::shared_ptr<const ov::Node>>& IExecutableNetworkInternal::getInputs() const {
    return _parameters;
}
void IExecutableNetworkInternal::setOutputs(const std::vector<std::shared_ptr<const ov::Node>>& results) {
    _results = results;
}
const std::vector<std::shared_ptr<const ov::Node>>& IExecutableNetworkInternal::getOutputs() const {
    return _results;
}

ConstOutputsDataMap IExecutableNetworkInternal::GetOutputsInfo() const {
    ConstOutputsDataMap outputMap;
    for (const auto& output : _networkOutputs) {
        outputMap.emplace(output.first, output.second);
    }
    return outputMap;
}

ConstInputsDataMap IExecutableNetworkInternal::GetInputsInfo() const {
    ConstInputsDataMap inputMap;
    for (const auto& input : _networkInputs) {
        inputMap.emplace(input.first, input.second);
    }
    return inputMap;
}

std::shared_ptr<IInferRequestInternal> IExecutableNetworkInternal::CreateInferRequest() {
    std::shared_ptr<IInferRequestInternal> asyncRequestImpl;
    try {
        asyncRequestImpl = CreateInferRequestImpl(_parameters, _results);
    } catch (const InferenceEngine::NotImplemented&) {
    }
    if (!asyncRequestImpl)
        asyncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    asyncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    return asyncRequestImpl;
}

void IExecutableNetworkInternal::Export(const std::string& modelFileName) {
    std::ofstream modelFile(modelFileName, std::ios::out | std::ios::binary);

    if (modelFile.is_open()) {
        Export(modelFile);
    } else {
        IE_THROW() << "The " << modelFileName << " file can not be opened for Export";
    }
}

void IExecutableNetworkInternal::Export(std::ostream& networkModel) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<ngraph::Function> IExecutableNetworkInternal::GetExecGraphInfo() {
    IE_THROW(NotImplemented);
}

void IExecutableNetworkInternal::SetPointerToPlugin(const std::shared_ptr<IInferencePlugin>& plugin) {
    _plugin = plugin;
}

void IExecutableNetworkInternal::SetConfig(const std::map<std::string, Parameter>& config) {
    _config = config;
}

Parameter IExecutableNetworkInternal::GetConfig(const std::string& key) const {
    const auto it = _config.find(key);
    if (it != _config.end()) {
        return it->second;
    }

    IE_THROW(NotFound) << key <<" not found in the IExecutableNetworkInternal config";
}

Parameter IExecutableNetworkInternal::GetMetric(const std::string&) const {
    IE_THROW(NotImplemented);
}

std::shared_ptr<RemoteContext> IExecutableNetworkInternal::GetContext() const {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IInferRequestInternal> IExecutableNetworkInternal::CreateInferRequestImpl(
    InputsDataMap networkInputs,
    OutputsDataMap networkOutputs) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IInferRequestInternal> IExecutableNetworkInternal::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    IE_THROW(NotImplemented);
}

}  // namespace InferenceEngine
