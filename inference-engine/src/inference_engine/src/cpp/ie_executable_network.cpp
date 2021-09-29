// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_executable_network.hpp"

#include "cpp/exception2status.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "ie_common.h"
#include "ie_executable_network_base.hpp"
#include "ie_remote_context.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/executable_network.hpp"

namespace InferenceEngine {

#define EXEC_NET_CALL_STATEMENT(...)                                        \
    if (_impl == nullptr)                                                   \
        IE_THROW(NotAllocated) << "ExecutableNetwork was not initialized."; \
    try {                                                                   \
        __VA_ARGS__;                                                        \
    } catch (...) {                                                         \
        InferenceEngine::details::Rethrow();                                \
    }

#define OV_EXEC_NET_CALL_STATEMENT(...)                                          \
    OPENVINO_ASSERT(_impl != nullptr, "ExecutableNetwork was not initialized."); \
    try {                                                                        \
        __VA_ARGS__;                                                             \
    } catch (const std::exception& ex) {                                         \
        throw ov::Exception(ex.what());                                          \
    } catch (...) {                                                              \
        OPENVINO_ASSERT(false, "Unexpected exception");                          \
    }

ExecutableNetwork::ExecutableNetwork(const details::SharedObjectLoader& so, const IExecutableNetworkInternal::Ptr& impl)
    : _so(so),
      _impl(impl) {
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

std::vector<VariableState> ExecutableNetwork::QueryState() {
    std::vector<VariableState> controller;
    EXEC_NET_CALL_STATEMENT({
        for (auto&& state : _impl->QueryState()) {
            controller.emplace_back(VariableState{_so, state});
        }
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
    EXEC_NET_CALL_STATEMENT(return CNNNetwork{_impl->GetExecGraphInfo()});
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

namespace ov {
namespace runtime {
ExecutableNetwork::ExecutableNetwork(const std::shared_ptr<void>& so,
                                     const std::shared_ptr<ie::IExecutableNetworkInternal>& impl)
    : _so{so},
      _impl{impl} {
    OPENVINO_ASSERT(_impl != nullptr, "ExecutableNetwork was not initialized.");
}

std::shared_ptr<const Function> ExecutableNetwork::get_runtime_function() const {
    OV_EXEC_NET_CALL_STATEMENT(return std::const_pointer_cast<const Function>(_impl->GetExecGraphInfo()));
}

std::vector<ov::Output<const ov::Node>> ExecutableNetwork::inputs() const {
    std::vector<ov::Output<const ov::Node>> inputs;
    for (const auto& input : _impl->getParameters()) {
        std::shared_ptr<const ov::Node> parameter = input;
        inputs.emplace_back(parameter);
    }
    return inputs;
}

ov::Output<const ov::Node> ExecutableNetwork::input() const {
    const auto params = _impl->getParameters();
    if (params.size() != 1) {
        throw ov::Exception("input() must be called on a function with exactly one parameter.");
    }
    return params.at(0);
}

ov::Output<const ov::Node> ExecutableNetwork::input(size_t i) const {
    OV_EXEC_NET_CALL_STATEMENT(return _impl->getParameters().at(i));
}

ov::Output<const ov::Node> ExecutableNetwork::input(const std::string& tensor_name) const {
    for (const auto& param : _impl->getParameters()) {
        if (param->get_output_tensor(0).get_names().count(tensor_name)) {
            return param;
        }
    }
    throw ov::Exception("Input for tensor name " + tensor_name + " was not found.");
}

std::vector<ov::Output<const ov::Node>> ExecutableNetwork::outputs() const {
    std::vector<ov::Output<const ov::Node>> outputs;
    for (const auto& input : _impl->getResults()) {
        std::shared_ptr<const ov::Node> result = input;
        outputs.emplace_back(result);
    }
    return outputs;
}
ov::Output<const ov::Node> ExecutableNetwork::output() const {
    const auto result = _impl->getResults();
    if (result.size() != 1) {
        throw ov::Exception("output() must be called on a function with exactly one parameter.");
    }
    return result.at(0);
}
ov::Output<const ov::Node> ExecutableNetwork::output(size_t i) const {
    OV_EXEC_NET_CALL_STATEMENT(return _impl->getResults().at(i));
}
ov::Output<const ov::Node> ExecutableNetwork::output(const std::string& tensor_name) const {
    for (const auto& result : _impl->getResults()) {
        if (result->get_output_tensor(0).get_names().count(tensor_name)) {
            return result;
        }
    }
    throw ov::Exception("Output for tensor name " + tensor_name + " was not found.");
}

InferRequest ExecutableNetwork::create_infer_request() {
    OV_EXEC_NET_CALL_STATEMENT(return {_so, _impl->CreateInferRequest()});
}

void ExecutableNetwork::export_model(std::ostream& networkModel) {
    OV_EXEC_NET_CALL_STATEMENT(_impl->Export(networkModel));
}

void ExecutableNetwork::set_config(const ie::ParamMap& config) {
    OV_EXEC_NET_CALL_STATEMENT(_impl->SetConfig(config));
}

ie::Parameter ExecutableNetwork::get_config(const std::string& name) const {
    OV_EXEC_NET_CALL_STATEMENT(return _impl->GetConfig(name));
}

ie::Parameter ExecutableNetwork::get_metric(const std::string& name) const {
    OV_EXEC_NET_CALL_STATEMENT(return _impl->GetMetric(name));
}

std::shared_ptr<ie::RemoteContext> ExecutableNetwork::get_context() const {
    OV_EXEC_NET_CALL_STATEMENT(return _impl->GetContext());
}

bool ExecutableNetwork::operator!() const noexcept {
    return !_impl;
}

ExecutableNetwork::operator bool() const noexcept {
    return !!_impl;
}
}  // namespace runtime
}  // namespace ov
