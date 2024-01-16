// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_executable_network.hpp"

#include "any_copy.hpp"
#include "cpp/exception2status.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "ie_common.h"
#include "ie_executable_network_base.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace InferenceEngine {

#define EXEC_NET_CALL_STATEMENT(...)                                        \
    if (_impl == nullptr)                                                   \
        IE_THROW(NotAllocated) << "ExecutableNetwork was not initialized."; \
    try {                                                                   \
        __VA_ARGS__;                                                        \
    } catch (...) {                                                         \
        InferenceEngine::details::Rethrow();                                \
    }

ExecutableNetwork::~ExecutableNetwork() {
    _impl = {};
}

ExecutableNetwork::ExecutableNetwork(const IExecutableNetworkInternal::Ptr& impl, const std::shared_ptr<void>& so)
    : _impl(impl),
      _so(so) {
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
    if (_impl == nullptr)
        IE_THROW() << "ExecutableNetwork was not initialized.";
    if (newActual == nullptr)
        IE_THROW() << "ExecutableNetwork wrapper used for reset was not initialized.";
    auto newBase = std::dynamic_pointer_cast<ExecutableNetworkBase>(newActual);
    IE_ASSERT(newBase != nullptr);
    auto newImpl = newBase->GetImpl();
    IE_ASSERT(newImpl != nullptr);
    _impl = newImpl;
}

ExecutableNetwork::operator IExecutableNetwork::Ptr() {
    return std::make_shared<ExecutableNetworkBase>(_impl);
}

InferRequest ExecutableNetwork::CreateInferRequest() {
    EXEC_NET_CALL_STATEMENT(return {_impl->CreateInferRequest(), _so});
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
    EXEC_NET_CALL_STATEMENT(return CNNNetwork{_impl->GetExecGraphInfo()});
}

void ExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    EXEC_NET_CALL_STATEMENT(_impl->SetConfig(config));
}

Parameter ExecutableNetwork::GetConfig(const std::string& name) const {
    EXEC_NET_CALL_STATEMENT(return {_impl->GetConfig(name), {_so}});
}

Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    EXEC_NET_CALL_STATEMENT(return {_impl->GetMetric(name), {_so}});
}

bool ExecutableNetwork::operator!() const noexcept {
    return !_impl;
}

ExecutableNetwork::operator bool() const noexcept {
    return !!_impl;
}
}  // namespace InferenceEngine
