// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_common.h"

#include "cpp/ie_executable_network.hpp"
#include "ie_executable_network_base.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"

namespace InferenceEngine {

#define EXEC_NET_CALL_STATEMENT(...)                                                               \
    if (_ptr == nullptr) IE_THROW() << "ExecutableNetwork was not initialized.";                   \
    try {                                                                                          \
        __VA_ARGS__;                                                                               \
    } CATCH_IE_EXCEPTIONS catch (const std::exception& ex) {                                       \
        IE_THROW() << ex.what();                                                                   \
    } catch (...) {                                                                                \
        IE_THROW(Unexpected);                                                                      \
    }

ExecutableNetwork::ExecutableNetwork() :
    details::SOPointer<IExecutableNetworkInternal>() { }

ExecutableNetwork::ExecutableNetwork(const details::SOPointer<IExecutableNetworkInternal> & obj) :
    details::SOPointer<IExecutableNetworkInternal>::SOPointer(obj) { }

ConstOutputsDataMap ExecutableNetwork::GetOutputsInfo() const {
    EXEC_NET_CALL_STATEMENT(return _ptr->GetOutputsInfo());
}

ConstInputsDataMap ExecutableNetwork::GetInputsInfo() const {
    EXEC_NET_CALL_STATEMENT(return _ptr->GetInputsInfo());
}

IE_SUPPRESS_DEPRECATED_START

void ExecutableNetwork::reset(IExecutableNetwork::Ptr newActual) {
    if (_ptr == nullptr) IE_THROW() << "ExecutableNetwork was not initialized.";
    if (newActual == nullptr) IE_THROW() << "ExecutableNetwork wrapper used for reset was not initialized.";
    auto newBase = std::dynamic_pointer_cast<ExecutableNetworkBase>(newActual);
    IE_ASSERT(newBase != nullptr);
    auto newImpl = newBase->GetImpl();
    IE_ASSERT(newImpl != nullptr);
    _ptr = details::SOPointer<IExecutableNetworkInternal>{_so, newImpl};
}

ExecutableNetwork::operator IExecutableNetwork::Ptr() {
    return std::make_shared<ExecutableNetworkBase>(_ptr);
}

std::vector<VariableState> ExecutableNetwork::QueryState() {
    std::vector<VariableState> controller;
    EXEC_NET_CALL_STATEMENT(
        for (auto&& state : _ptr->QueryState()) {
            controller.emplace_back(_so, state);
        });
    return controller;
}

IE_SUPPRESS_DEPRECATED_END

InferRequest ExecutableNetwork::CreateInferRequest() {
    EXEC_NET_CALL_STATEMENT(return InferRequest{_so, _ptr->CreateInferRequest()});
}

InferRequest::Ptr ExecutableNetwork::CreateInferRequestPtr() {
    EXEC_NET_CALL_STATEMENT(return std::make_shared<InferRequest>(_so, _ptr->CreateInferRequest()));
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    EXEC_NET_CALL_STATEMENT(_ptr->Export(modelFileName));
}

void ExecutableNetwork::Export(std::ostream& networkModel) {
    EXEC_NET_CALL_STATEMENT(_ptr->Export(networkModel));
}

CNNNetwork ExecutableNetwork::GetExecGraphInfo() {
    EXEC_NET_CALL_STATEMENT(return _ptr->GetExecGraphInfo());
}

void ExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    EXEC_NET_CALL_STATEMENT(_ptr->SetConfig(config));
}

Parameter ExecutableNetwork::GetConfig(const std::string& name) const {
    EXEC_NET_CALL_STATEMENT(return _ptr->GetConfig(name));
}

Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    EXEC_NET_CALL_STATEMENT(return _ptr->GetMetric(name));
}

RemoteContext::Ptr ExecutableNetwork::GetContext() const {
    EXEC_NET_CALL_STATEMENT(return _ptr->GetContext());
}

bool ExecutableNetwork::operator!() const noexcept {
    return !_ptr;
}

ExecutableNetwork::operator bool() const noexcept {
    return !!_ptr;
}
}  // namespace InferenceEngine
