// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <map>

#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include <ie_plugin_config.hpp>

#include "utils/log_util.hpp"

#include "itt.hpp"
#include "base_executable_network.hpp"
// ------------------------------BaseExecutableNetwork----------------------------
namespace MultiDevicePlugin {
using namespace InferenceEngine;

BaseExecutableNetwork::BaseExecutableNetwork(const Schedule::Ptr& schedule,
    const Context::Ptr& context):
    _schedule(schedule),
    _context(context) {
    //_executableNetwork = _schedule->GetExecNetwork();
}
BaseExecutableNetwork::~BaseExecutableNetwork() {
    _schedule->release();
}

std::shared_ptr<InferenceEngine::RemoteContext>
BaseExecutableNetwork::GetContext() const {
    return _executableNetwork->GetContext();
}

IInferPtr BaseExecutableNetwork::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    return _schedule->CreateInferRequestImpl(inputs, outputs);
}

IInferPtr BaseExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    return _schedule->CreateInferRequestImpl(networkInputs, networkOutputs);
}

IInferRequestInternal::Ptr BaseExecutableNetwork::CreateInferRequest() {
    SetExeNetworkForContext();
    return _schedule->CreateInferRequest();
}

void BaseExecutableNetwork::SetConfig(const
    std::map<std::string, InferenceEngine::Parameter>& config) {
    return _executableNetwork->SetConfig(config);
}

InferenceEngine::Parameter BaseExecutableNetwork::GetConfig(
    const std::string& name) const {
    return _executableNetwork->GetConfig(name);
}

InferenceEngine::Parameter BaseExecutableNetwork::GetMetric(
    const std::string& name) const {
    return _executableNetwork->GetMetric(name);
}

void BaseExecutableNetwork::SetExeNetworkForContext() {
    // Maybe different API will call this function, so add call once here
    // for every AutoSchedule instance
    std::call_once(_oc, [this]() {
        _context->_executableNetwork = shared_from_this();
    });
}
}  // namespace MultiDevicePlugin
