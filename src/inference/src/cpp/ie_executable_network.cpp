// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_executable_network.hpp"

#include "any_copy.hpp"
#include "cpp/exception2status.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "ie_common.h"
#include "ie_executable_network_base.hpp"
#include "ie_plugin_config.hpp"
#include "ie_remote_context.hpp"
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

#define OV_EXEC_NET_CALL_STATEMENT(...)                                          \
    OPENVINO_ASSERT(_impl != nullptr, "ExecutableNetwork was not initialized."); \
    try {                                                                        \
        __VA_ARGS__;                                                             \
    } catch (const std::exception& ex) {                                         \
        throw ov::Exception(ex.what());                                          \
    } catch (...) {                                                              \
        OPENVINO_ASSERT(false, "Unexpected exception");                          \
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
    EXEC_NET_CALL_STATEMENT(return {_impl->GetConfig(name), _so});
}

Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    EXEC_NET_CALL_STATEMENT(return {_impl->GetMetric(name), _so});
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

CompiledModel::~CompiledModel() {
    _impl = {};
}

CompiledModel::CompiledModel(const std::shared_ptr<ie::IExecutableNetworkInternal>& impl,
                             const std::shared_ptr<void>& so)
    : _impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "CompiledModel was not initialized.");
}

std::shared_ptr<const Model> CompiledModel::get_runtime_model() const {
    OV_EXEC_NET_CALL_STATEMENT(return std::const_pointer_cast<const Model>(_impl->GetExecGraphInfo()));
}

std::vector<ov::Output<const ov::Node>> CompiledModel::inputs() const {
    OV_EXEC_NET_CALL_STATEMENT({
        std::vector<ov::Output<const ov::Node>> inputs;
        for (const auto& input : _impl->getInputs()) {
            inputs.emplace_back(input);
        }
        return inputs;
    });
}

ov::Output<const ov::Node> CompiledModel::input() const {
    OV_EXEC_NET_CALL_STATEMENT({
        const auto inputs = _impl->getInputs();
        if (inputs.size() != 1) {
            throw ov::Exception("input() must be called on a function with exactly one parameter.");
        }
        return inputs.at(0);
    });
}

ov::Output<const ov::Node> CompiledModel::input(size_t i) const {
    OV_EXEC_NET_CALL_STATEMENT(return _impl->getInputs().at(i));
}

ov::Output<const ov::Node> CompiledModel::input(const std::string& tensor_name) const {
    OV_EXEC_NET_CALL_STATEMENT({
        for (const auto& param : _impl->getInputs()) {
            if (param->get_output_tensor(0).get_names().count(tensor_name)) {
                return param;
            }
        }
        throw ov::Exception("Input for tensor name '" + tensor_name + "' is not found.");
    });
}

std::vector<ov::Output<const ov::Node>> CompiledModel::outputs() const {
    OV_EXEC_NET_CALL_STATEMENT({
        std::vector<ov::Output<const ov::Node>> outputs;
        for (const auto& output : _impl->getOutputs()) {
            outputs.emplace_back(output);
        }
        return outputs;
    });
}
ov::Output<const ov::Node> CompiledModel::output() const {
    OV_EXEC_NET_CALL_STATEMENT({
        const auto outputs = _impl->getOutputs();
        if (outputs.size() != 1) {
            throw ov::Exception("output() must be called on a function with exactly one result.");
        }
        return outputs.at(0);
    });
}
ov::Output<const ov::Node> CompiledModel::output(size_t i) const {
    OV_EXEC_NET_CALL_STATEMENT(return _impl->getOutputs().at(i));
}
ov::Output<const ov::Node> CompiledModel::output(const std::string& tensor_name) const {
    OV_EXEC_NET_CALL_STATEMENT({
        for (const auto& result : _impl->getOutputs()) {
            if (result->get_output_tensor(0).get_names().count(tensor_name)) {
                return result;
            }
        }
        throw ov::Exception("Output for tensor name '" + tensor_name + "' is not found.");
    });
}

InferRequest CompiledModel::create_infer_request() {
    OV_EXEC_NET_CALL_STATEMENT(return {_impl->CreateInferRequest(), _so});
}

void CompiledModel::export_model(std::ostream& networkModel) {
    OV_EXEC_NET_CALL_STATEMENT(_impl->Export(networkModel));
}

void CompiledModel::set_property(const AnyMap& config) {
    OV_EXEC_NET_CALL_STATEMENT(_impl->SetConfig(config));
}

Any CompiledModel::get_property(const std::string& name) const {
    OV_EXEC_NET_CALL_STATEMENT({
        if (ov::supported_properties == name) {
            try {
                auto supported_properties = _impl->GetMetric(name).as<std::vector<PropertyName>>();
                supported_properties.erase(std::remove_if(supported_properties.begin(),
                                                          supported_properties.end(),
                                                          [](const ov::PropertyName& name) {
                                                              return name == METRIC_KEY(SUPPORTED_METRICS) ||
                                                                     name == METRIC_KEY(SUPPORTED_CONFIG_KEYS);
                                                          }),
                                           supported_properties.end());
                return supported_properties;
            } catch (ie::Exception&) {
                auto ro_properties = _impl->GetMetric(METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
                auto rw_properties = _impl->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
                std::vector<ov::PropertyName> supported_properties;
                for (auto&& ro_property : ro_properties) {
                    if (ro_property != METRIC_KEY(SUPPORTED_METRICS) &&
                        ro_property != METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                        supported_properties.emplace_back(ro_property, PropertyMutability::RO);
                    }
                }
                for (auto&& rw_property : rw_properties) {
                    supported_properties.emplace_back(rw_property, PropertyMutability::RW);
                }
                supported_properties.emplace_back(ov::supported_properties.name(), PropertyMutability::RO);
                return supported_properties;
            }
        }
        try {
            return {_impl->GetMetric(name), _so};
        } catch (ie::Exception&) {
            return {_impl->GetConfig(name), _so};
        }
    });
}

RemoteContext CompiledModel::get_context() const {
    OV_EXEC_NET_CALL_STATEMENT(return {_impl->GetContext(), _so});
}

bool CompiledModel::operator!() const noexcept {
    return !_impl;
}

CompiledModel::operator bool() const noexcept {
    return !!_impl;
}

}  // namespace ov
