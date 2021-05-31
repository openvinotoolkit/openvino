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

AutoExecutableNetwork::AutoExecutableNetwork(NetworkFuture cpuFuture,
                                             NetworkFuture acceleratorFuture,
                                             bool          enablePerfCount)
                                             : _cpuFuture(std::move(cpuFuture))
                                             , _acceleratorFuture(std::move(acceleratorFuture))
                                             , _enablePerfCount(enablePerfCount) {
    // we wait for any network to become ready (maybe this will already an actual device)
    if (_cpuFuture.valid()) {
        _networkFirstReady = _cpuFuture.get();
    } else if (_acceleratorFuture.valid()) {
        _networkActualNeeded = _acceleratorFuture.get();
        _alreadyActualNetwork = true;
    } else {
        IE_THROW() << "No device task available";
    }
}

AutoExecutableNetwork::~AutoExecutableNetwork() = default;

InferenceEngine::IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                                          OutputsDataMap networkOutputs) {
    TryGetActualNetwork(_networkActualNeeded);

    SoIInferRequestInternal inferRequest;
    if (_alreadyActualNetwork) {
        inferRequest = {_networkActualNeeded, _networkActualNeeded->CreateInferRequest()};
    } else {
        inferRequest = {_networkFirstReady, _networkFirstReady->CreateInferRequest()};
    }
    return std::make_shared<AutoInferRequest>(_networkInputs, _networkOutputs, inferRequest,
                                              shared_from_this(), _alreadyActualNetwork,
                                              _enablePerfCount);
}

bool AutoExecutableNetwork::TryGetActualNetwork(InferenceEngine::SoExecutableNetworkInternal& soExecNetwork) {
    if (_acceleratorFuture.valid() && _acceleratorFuture.wait_for(std::chrono::nanoseconds(0)) == std::future_status::ready) {
        soExecNetwork = _acceleratorFuture.get();
        _alreadyActualNetwork = true;
        _networkActualNeeded = soExecNetwork;
        return true;
    }
    if (_alreadyActualNetwork) {
        soExecNetwork = _networkActualNeeded;
        return true;
    }
    return false;
}

void AutoExecutableNetwork::Export(std::ostream& networkModel) {
    //fixme: the Export  should work with actual device, so we have to wait!!!
//    wait_for_actual_device();
//    _networkActualNeeded->Export(networkModel);
}

RemoteContext::Ptr AutoExecutableNetwork::GetContext() const {
    // fixme: the GetContext  should work with actual device, so we have to wait!!!
//   wait_for_actual_device();
//   return (_networkActualNeeded) ? _networkActualNeeded->GetContext() : RemoteContext::Ptr{};
     return RemoteContext::Ptr{};
}

InferenceEngine::CNNNetwork AutoExecutableNetwork::GetExecGraphInfo() {
    // fixme: still not safe - shoujiang
    return _alreadyActualNetwork ? _networkActualNeeded->GetExecGraphInfo() : _networkFirstReady->GetExecGraphInfo();
}

Parameter AutoExecutableNetwork::GetMetric(const std::string &name) const {
    //fixme: check this logic
    return _alreadyActualNetwork ? _networkActualNeeded->GetMetric(name) : _networkFirstReady->GetMetric(name);
}

void AutoExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
     //fixme: have to store the config and reapply when the networks swapped
    _networkFirstReady->SetConfig(config);
}

Parameter AutoExecutableNetwork::GetConfig(const std::string& name) const {
    //fixme: carefuly select between FirstLoaded and ActuallyNeeded
    return _networkFirstReady->GetConfig(name);
}

}  // namespace AutoPlugin
