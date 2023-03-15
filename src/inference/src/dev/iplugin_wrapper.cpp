// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "iplugin_wrapper.hpp"

#include <memory>

#include "any_copy.hpp"
#include "dev/converter_utils.hpp"
#include "ie_icore.hpp"
#include "threading/ie_executor_manager.hpp"

namespace InferenceEngine {

IPluginWrapper::IPluginWrapper(const std::shared_ptr<InferenceEngine::IInferencePlugin>& ptr) : m_old_plugin(ptr) {
    OPENVINO_ASSERT(m_old_plugin);
    auto& ver = m_old_plugin->GetVersion();
    m_version.buildNumber = ver.buildNumber;
    m_version.description = ver.description;
    m_plugin_name = m_old_plugin->GetName();
    m_is_new_api = m_old_plugin->IsNewAPI();
    m_core = m_old_plugin->GetCore();
    m_executor_manager = m_old_plugin->executorManager()->get_ov_manager();
}

const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& IPluginWrapper::update_exec_network(
    const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& network) const {
    network->SetPointerToPlugin(m_old_plugin);
    return network;
}

std::shared_ptr<ov::ICompiledModel> IPluginWrapper::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                  const ov::AnyMap& properties) const {
    auto exec_network =
        m_old_plugin->LoadNetwork(ov::legacy_convert::convert_model(model, is_new_api()), ov::any_copy(properties));
    return ov::legacy_convert::convert_compiled_model(update_exec_network(exec_network));
}

std::shared_ptr<ov::ICompiledModel> IPluginWrapper::compile_model(const std::string& model_path,
                                                                  const ov::AnyMap& properties) const {
    auto exec_network = m_old_plugin->LoadNetwork(model_path, any_copy(properties));
    return ov::legacy_convert::convert_compiled_model(update_exec_network(exec_network._ptr));
}

std::shared_ptr<ov::ICompiledModel> IPluginWrapper::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                  const ov::AnyMap& properties,
                                                                  const ov::RemoteContext& context) const {
    return ov::legacy_convert::convert_compiled_model(
        update_exec_network(m_old_plugin->LoadNetwork(ov::legacy_convert::convert_model(model, is_new_api()),
                                                      any_copy(properties),
                                                      context._impl)));
}

void IPluginWrapper::set_property(const ov::AnyMap& properties) {
    m_old_plugin->SetProperties(properties);
}

ov::Any IPluginWrapper::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    try {
        return m_old_plugin->GetConfig(name, arguments);
    } catch (...) {
        return m_old_plugin->GetMetric(name, arguments);
    }
}

ov::RemoteContext IPluginWrapper::create_context(const ov::AnyMap& remote_properties) const {
    return ov::RemoteContext{m_old_plugin->CreateContext(remote_properties), {nullptr}};
}

ov::RemoteContext IPluginWrapper::get_default_context(const ov::AnyMap& remote_properties) const {
    return ov::RemoteContext{m_old_plugin->GetDefaultContext(remote_properties), {nullptr}};
}

std::shared_ptr<ov::ICompiledModel> IPluginWrapper::import_model(std::istream& model,
                                                                 const ov::AnyMap& properties) const {
    return ov::legacy_convert::convert_compiled_model(
        update_exec_network(m_old_plugin->ImportNetwork(model, any_copy(properties))));
}

std::shared_ptr<ov::ICompiledModel> IPluginWrapper::import_model(std::istream& model,
                                                                 const ov::RemoteContext& context,
                                                                 const ov::AnyMap& properties) const {
    return ov::legacy_convert::convert_compiled_model(
        update_exec_network(m_old_plugin->ImportNetwork(model, context._impl, any_copy(properties))));
}

ov::SupportedOpsMap IPluginWrapper::query_model(const std::shared_ptr<const ov::Model>& model,
                                                const ov::AnyMap& properties) const {
    auto res = m_old_plugin->QueryNetwork(ov::legacy_convert::convert_model(model, is_new_api()), any_copy(properties));
    OPENVINO_ASSERT(res.rc == InferenceEngine::OK, res.resp.msg);
    return res.supportedLayersMap;
}

void IPluginWrapper::add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension) {
    m_old_plugin->AddExtension(extension);
}

const std::shared_ptr<InferenceEngine::IInferencePlugin>& IPluginWrapper::get_plugin() const {
    return m_old_plugin;
}

void IPluginWrapper::set_core(const std::weak_ptr<ov::ICore>& core) {
    auto locked_core = core.lock();
    auto old_core = std::dynamic_pointer_cast<InferenceEngine::ICore>(locked_core);
    if (old_core)
        m_old_plugin->SetCore(old_core);
    m_core = core;
}

void IPluginWrapper::set_device_name(const std::string& device_name) {
    m_plugin_name = device_name;
    m_old_plugin->SetName(device_name);
}

}  //  namespace InferenceEngine
