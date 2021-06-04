// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>

#include "ie_metric_helpers.hpp"
#include "auto_exec_network.hpp"
#include "auto_infer_request.hpp"

namespace AutoPlugin {
using namespace InferenceEngine;

AutoExecutableNetwork::AutoExecutableNetwork(NetworkTaskSharedPtr cpuTask,
                                             NetworkTaskSharedPtr acceleratorTask,
                                             InferenceEngine::IStreamsExecutor::Ptr cpuExecutor)
                                             : _cpuExecutor(cpuExecutor) {
    // we wait for any network to become ready (maybe this will already an actual device)
    if (cpuTask) {
        _networkFirstReady = cpuTask->get_future().get();
    } else if (acceleratorTask) {
        _networkFirstReady = acceleratorTask->get_future().get();
    } else {
        IE_THROW() << "No device task available";
    }

    if (acceleratorTask) {
        try {
            _sharedFutureActualNetwork = acceleratorTask->get_future().share();
        } catch (const std::future_error& e) {
        }
    }
}

AutoExecutableNetwork::~AutoExecutableNetwork() {
    try {
        if (_sharedFutureActualNetwork.valid()) {
            _sharedFutureActualNetwork.get();
        }
    } catch (...) {
    }
}

InferenceEngine::IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                                          OutputsDataMap networkOutputs) {
    // fixme: may need consider swap too. - shoujiang
    SoIInferRequestInternal inferRequest = {_networkFirstReady, _networkFirstReady->CreateInferRequest()};
    return std::make_shared<AutoInferRequest>(_networkInputs, _networkOutputs, inferRequest,
        _sharedFutureActualNetwork, _anyRequestHasHotSwapped);
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
    return _anyRequestHasHotSwapped ? _networkActualNeeded->GetExecGraphInfo() : _networkFirstReady->GetExecGraphInfo();
}

Parameter AutoExecutableNetwork::GetMetric(const std::string &name) const {
    //fixme: check this logic
    return _anyRequestHasHotSwapped ? _networkActualNeeded->GetMetric(name) : _networkFirstReady->GetMetric(name);
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
