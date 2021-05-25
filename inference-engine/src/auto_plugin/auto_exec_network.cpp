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

AutoExecutableNetwork::AutoExecutableNetwork(AutoPlugin::NetworkPromiseSharedPtr networkPromiseFirstReady,
                                             AutoPlugin::NetworkPromiseSharedPtr networkPromiseActualNeeded) {
    // we wait for any network to become ready (maybe this will already an actual device)
    _networkFirstReady = networkPromiseFirstReady->get_future().get();
    _networkPromiseActualNeeded = networkPromiseActualNeeded;
    futureActualNetwork = _networkPromiseActualNeeded->get_future();
}

AutoExecutableNetwork::~AutoExecutableNetwork() = default;

InferenceEngine::IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                                          OutputsDataMap networkOutputs) {
    SoIInferRequestInternal inferRequest = {_networkFirstReady, _networkFirstReady->CreateInferRequest()};
    return std::make_shared<AutoInferRequest>(_networkInputs, _networkOutputs, inferRequest, futureActualNetwork.share());
}

void AutoExecutableNetwork::Export(std::ostream& networkModel) {
    wait_for_actual_device();
    _networkActualNeeded->Export(networkModel);
}

RemoteContext::Ptr AutoExecutableNetwork::GetContext() const {
   wait_for_actual_device();
   return _networkActualNeeded->GetContext();
}

InferenceEngine::CNNNetwork AutoExecutableNetwork::GetExecGraphInfo() {
    wait_for_actual_device();
    return _networkFirstReady->GetExecGraphInfo();
}

Parameter AutoExecutableNetwork::GetMetric(const std::string &name) const {
    return _networkFirstReady->GetMetric(name);
}

void AutoExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    // this seems to be Not GOOD, why should we have SetConfig for the AUTO? AUTO has no config options
    // _networkFirstReady->SetConfig(config);
}

Parameter AutoExecutableNetwork::GetConfig(const std::string& name) const {
    // fixme: also change to the FirstLoaded vs ActuallyNeeeded
    return {};//  _networkFirstReady->GetConfig(name);
}

}  // namespace AutoPlugin
