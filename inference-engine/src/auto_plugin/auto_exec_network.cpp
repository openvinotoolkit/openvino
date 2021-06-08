// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>
#include <map>

#include "ie_metric_helpers.hpp"
#include "auto_exec_network.hpp"
#include "auto_infer_request.hpp"

namespace AutoPlugin {
using namespace InferenceEngine;

AutoExecutableNetwork::AutoExecutableNetwork(const SoExecutableNetworkInternal& network, bool enablePerfCount) :
    _network(network), _enablePerfCount(enablePerfCount) {
}

AutoExecutableNetwork::~AutoExecutableNetwork() = default;

InferenceEngine::IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                                          OutputsDataMap networkOutputs) {
    SoIInferRequestInternal inferRequest = {_network, _network->CreateInferRequest()};
    return std::make_shared<AutoInferRequest>(_networkInputs, _networkOutputs, inferRequest, _enablePerfCount);
}

void AutoExecutableNetwork::Export(std::ostream& networkModel) {
    _network->Export(networkModel);
}

RemoteContext::Ptr AutoExecutableNetwork::GetContext() const {
  return _network->GetContext();
}

InferenceEngine::CNNNetwork AutoExecutableNetwork::GetExecGraphInfo() {
    return _network->GetExecGraphInfo();
}

Parameter AutoExecutableNetwork::GetMetric(const std::string &name) const {
    return _network->GetMetric(name);
}

void AutoExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    _network->SetConfig(config);
}

Parameter AutoExecutableNetwork::GetConfig(const std::string& name) const {
    return _network->GetConfig(name);
}

}  // namespace AutoPlugin
