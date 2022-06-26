// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "executable_network.hpp"
// ------------------------------ExecutableNetwork----------------------------
namespace MultiDevicePlugin {
using namespace InferenceEngine;

ExecutableNetwork::ExecutableNetwork(const Schedule::Ptr& schedule,
    const ScheduleContext::Ptr& sContext):
    _schedule(schedule),
    _sContext(sContext) {
    _schedule->init(_sContext);
}

ExecutableNetwork::~ExecutableNetwork() {
}

IInferPtr ExecutableNetwork::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    return _schedule->CreateInferRequestImpl(inputs, outputs);
}

IInferPtr ExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    return _schedule->CreateInferRequestImpl(networkInputs, networkOutputs);
}

IInferRequestInternal::Ptr ExecutableNetwork::CreateInferRequest() {
    SetExeNetworkForContext();
    return _schedule->CreateInferRequest();
}

void ExecutableNetwork::SetExeNetworkForContext() {
    // Maybe different API will call this function, so add call once here
    // for every AutoSchedule instance
    std::call_once(_oc, [this]() {
        _sContext->_executableNetwork = shared_from_this();
    });
}

std::string ExecutableNetwork::GetLogTag() const noexcept {
    return _sContext->_LogTag;
}
}  // namespace MultiDevicePlugin
