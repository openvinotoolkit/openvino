// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_executable_network.hpp"
#include "ie_common.h"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "ie_iexecutable_network.hpp"
#include "cpp_interfaces/base/ie_executable_network_base.hpp"

namespace InferenceEngine {

#define CALL_STATEMENT(...)                                                                        \
    if (_impl == nullptr) IE_THROW() << "ExecutableNetwork was not initialized.";                  \
    try {                                                                                          \
        __VA_ARGS__;                                                                               \
    } CATCH_IE_EXCEPTIONS catch (const std::exception& ex) {                                       \
        IE_THROW() << ex.what();                                                                   \
    } catch (...) {                                                                                \
        IE_THROW(Unexpected);                                                                      \
    }

ExecutableNetwork::ExecutableNetwork(const IExecutableNetworkInternal::Ptr& impl,
                                     const std::shared_ptr<details::SharedObjectLoader>& so)
    : _impl(impl), _so(so) {
    IE_ASSERT(_impl != nullptr);
}

ExecutableNetwork::~ExecutableNetwork() {
    _impl = {};
}

ConstOutputsDataMap ExecutableNetwork::GetOutputsInfo() const {
    CALL_STATEMENT(return _impl->GetOutputsInfo());
}

ConstInputsDataMap ExecutableNetwork::GetInputsInfo() const {
    CALL_STATEMENT(return _impl->GetInputsInfo());
}

void ExecutableNetwork::reset(IExecutableNetwork::Ptr newActual) {
    if (_impl == nullptr) IE_THROW() << "ExecutableNetwork was not initialized.";
    if (newActual == nullptr) IE_THROW() << "ExecutableNetwork wrapper used for reset was not initialized.";
    auto newBase = std::dynamic_pointer_cast<ExecutableNetworkBase>(newActual);
    IE_ASSERT(newBase != nullptr);
    auto newImpl = newBase->GetImpl();
    IE_ASSERT(newImpl != nullptr);
    this->_impl.swap(newImpl);
}

InferRequest ExecutableNetwork::CreateInferRequest() {
    CALL_STATEMENT(return InferRequest{_impl->CreateInferRequest(), _so});
}

InferRequest::Ptr ExecutableNetwork::CreateInferRequestPtr() {
    CALL_STATEMENT(return std::make_shared<InferRequest>(_impl->CreateInferRequest(), _so));
}

void ExecutableNetwork::Export(const std::string& modelFileName) {
    CALL_STATEMENT(return _impl->Export(modelFileName));
}

void ExecutableNetwork::Export(std::ostream& networkModel) {
    CALL_STATEMENT(return _impl->Export(networkModel));
}

ExecutableNetwork::operator IExecutableNetwork::Ptr() {
    return std::make_shared<ExecutableNetworkBase>(_impl);
}

CNNNetwork ExecutableNetwork::GetExecGraphInfo() {
    IE_SUPPRESS_DEPRECATED_START
    CALL_STATEMENT(return _impl->GetExecGraphInfo());
}

IE_SUPPRESS_DEPRECATED_START
std::vector<VariableState> ExecutableNetwork::QueryState() {
    std::vector<VariableState> controller;
    CALL_STATEMENT(
        for (auto&& state : _impl->QueryState()) {
            controller.emplace_back(std::make_shared<VariableStateBase>(state), _so);
        });
    return controller;
}
IE_SUPPRESS_DEPRECATED_END

void ExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    CALL_STATEMENT(_impl->SetConfig(config));
}

Parameter ExecutableNetwork::GetConfig(const std::string& name) const {
    CALL_STATEMENT(return _impl->GetConfig(name));
}

Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    CALL_STATEMENT(return _impl->GetMetric(name));
}

RemoteContext::Ptr ExecutableNetwork::GetContext() const {
    CALL_STATEMENT(return _impl->GetContext());
}

bool ExecutableNetwork::operator!() const noexcept {
    return !_impl;
}

ExecutableNetwork::operator bool() const noexcept {
    return !!_impl;
}

}  // namespace InferenceEngine
