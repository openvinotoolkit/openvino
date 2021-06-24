// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_common.h"

#include "cpp/ie_executable_network.hpp"
#include "cpp/exception2status.hpp"
#include "ie_executable_network_base.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"

namespace InferenceEngine {

#define EXEC_NET_CALL_STATEMENT(...)                                                               \
    if (_impl == nullptr) IE_THROW(NotAllocated) << "ExecutableNetwork was not initialized.";      \
    try {                                                                                          \
        __VA_ARGS__;                                                                               \
    } catch(...) {details::Rethrow();}

ExecutableNetwork::ExecutableNetwork(const details::SharedObjectLoader&      so,
                                     const IExecutableNetworkInternal::Ptr&  impl)
    : _so(so), _impl(impl) {
    IE_ASSERT(_impl != nullptr);
}

IE_SUPPRESS_DEPRECATED_START

ConstOutputsDataMap ExecutableNetwork::GetOutputsInfo() const {
    EXEC_NET_CALL_STATEMENT(return _impl->GetOutputsInfo());
}

ConstInputsDataMap ExecutableNetwork::GetInputsInfo() const {
    EXEC_NET_CALL_STATEMENT(return _impl->GetInputsInfo());
}

void ExecutableNetwork::reset(IExecutableNetwork::Ptr newActual) {
    if (_impl == nullptr) IE_THROW() << "ExecutableNetwork was not initialized.";
    if (newActual == nullptr) IE_THROW() << "ExecutableNetwork wrapper used for reset was not initialized.";
    auto newBase = std::dynamic_pointer_cast<ExecutableNetworkBase>(newActual);
    IE_ASSERT(newBase != nullptr);
    auto newImpl = newBase->GetImpl();
    IE_ASSERT(newImpl != nullptr);
    _impl = newImpl;
}

ExecutableNetwork::operator IExecutableNetwork::Ptr() {
    return std::make_shared<ExecutableNetworkBase>(_impl);
}

std::vector<VariableState> ExecutableNetwork::QueryState() {
    std::vector<VariableState> controller;
    EXEC_NET_CALL_STATEMENT(
        for (auto&& state : _impl->QueryState()) {
            controller.emplace_back(VariableState{ _so, state });
        });
    return controller;
}

InferRequest ExecutableNetwork::CreateInferRequest() {
    EXEC_NET_CALL_STATEMENT(return {_so, _impl->CreateInferRequest()});
}

InferRequest::Ptr ExecutableNetwork::CreateInferRequestPtr() {
    return std::make_shared<InferRequest>(CreateInferRequest());
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    EXEC_NET_CALL_STATEMENT(_impl->Export(modelFileName));
}

void ExecutableNetwork::Export(std::ostream& networkModel) {
    EXEC_NET_CALL_STATEMENT(_impl->Export(networkModel));
}

CNNNetwork ExecutableNetwork::GetExecGraphInfo() {
    EXEC_NET_CALL_STATEMENT(return _impl->GetExecGraphInfo());
}

void ExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    EXEC_NET_CALL_STATEMENT(_impl->SetConfig(config));
}

Parameter ExecutableNetwork::GetConfig(const std::string& name) const {
    EXEC_NET_CALL_STATEMENT(return _impl->GetConfig(name));
}

Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    EXEC_NET_CALL_STATEMENT(return _impl->GetMetric(name));
}

RemoteContext::Ptr ExecutableNetwork::GetContext() const {
    EXEC_NET_CALL_STATEMENT(return _impl->GetContext());
}

bool ExecutableNetwork::operator!() const noexcept {
    return !_impl;
}

ExecutableNetwork::operator bool() const noexcept {
    return !!_impl;
}
}  // namespace InferenceEngine
