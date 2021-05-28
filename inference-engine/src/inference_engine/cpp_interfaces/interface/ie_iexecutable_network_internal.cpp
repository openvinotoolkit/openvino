// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

#include <ie_icore.hpp>
#include <ie_parameter.hpp>
#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {

void IExecutableNetworkInternal::setNetworkInputs(const InputsDataMap& networkInputs) {
    _networkInputs = networkInputs;
}

void IExecutableNetworkInternal::setNetworkOutputs(const OutputsDataMap& networkOutputs) {
    _networkOutputs = networkOutputs;
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
    auto asyncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    asyncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    return asyncRequestImpl;
}

void IExecutableNetworkInternal::Export(const std::string& modelFileName) {
    // we need to write to stringstream first
    // because in case of exception in ExportImpl the file is not created
    std::stringstream strm;
    ExportImpl(strm);
    std::ofstream(modelFileName.c_str()) << strm.rdbuf();
}

void IExecutableNetworkInternal::Export(std::ostream& networkModel) {
    std::stringstream strm;
    strm.write(exportMagic.data(), exportMagic.size());
    strm << _plugin->GetName() << std::endl;
    ExportImpl(strm);
    networkModel << strm.rdbuf();
}

CNNNetwork IExecutableNetworkInternal::GetExecGraphInfo() {
    IE_THROW(NotImplemented);
}

std::vector<std::shared_ptr<IVariableStateInternal>> IExecutableNetworkInternal::QueryState() {
    IE_THROW(NotImplemented);
}

void IExecutableNetworkInternal::SetPointerToPlugin(const std::shared_ptr<IInferencePlugin>& plugin) {
    _plugin = plugin;
}

void IExecutableNetworkInternal::SetConfig(const std::map<std::string, Parameter>&) {
    IE_THROW(NotImplemented);
}

Parameter IExecutableNetworkInternal::GetConfig(const std::string&) const {
    IE_THROW(NotImplemented);
}

Parameter IExecutableNetworkInternal::GetMetric(const std::string&) const {
    IE_THROW(NotImplemented);
}

std::shared_ptr<RemoteContext> IExecutableNetworkInternal::GetContext() const {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IInferRequestInternal> IExecutableNetworkInternal::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                                          OutputsDataMap networkOutputs) {
    IE_THROW(NotImplemented);
}

void IExecutableNetworkInternal::ExportImpl(std::ostream&) {
    IE_THROW(NotImplemented);
}
}  // namespace InferenceEngine
