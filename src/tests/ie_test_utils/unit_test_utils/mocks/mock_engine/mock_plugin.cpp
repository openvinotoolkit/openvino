// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugin.hpp"

#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <utility>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "description_buffer.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/iplugin.hpp"

MockPlugin::MockPlugin(ov::IPlugin* target) {
    m_plugin = target;
}

MockPlugin::MockPlugin(InferenceEngine::IInferencePlugin* target) {
    std::shared_ptr<InferenceEngine::IInferencePlugin> shared_target(target, [](InferenceEngine::IInferencePlugin*) {});
    m_converted_plugin = InferenceEngine::convert_plugin(shared_target);
    m_plugin = m_converted_plugin.get();
}
void MockPlugin::set_property(const ov::AnyMap& properties) {
    this->config = properties;
    if (m_plugin) {
        m_plugin->set_property(config);
    }
}

ov::Any MockPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (m_plugin)
        return m_plugin->get_property(name, arguments);
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> MockPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties) const {
    if (m_plugin)
        return m_plugin->compile_model(model, properties);
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> MockPlugin::compile_model(const std::string& model_path,
                                                              const ov::AnyMap& properties) const {
    if (m_plugin)
        return m_plugin->compile_model(model_path, properties);
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> MockPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties,
                                                              const ov::RemoteContext& context) const {
    if (m_plugin)
        return m_plugin->compile_model(model, properties, context);
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> MockPlugin::create_context(const ov::AnyMap& remote_properties) const {
    if (m_plugin)
        return m_plugin->create_context(remote_properties);
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> MockPlugin::get_default_context(const ov::AnyMap& remote_properties) const {
    if (m_plugin)
        return m_plugin->get_default_context(remote_properties);
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> MockPlugin::import_model(std::istream& model, const ov::AnyMap& properties) const {
    if (m_plugin)
        return m_plugin->import_model(model, properties);
    OPENVINO_NOT_IMPLEMENTED;
}
std::shared_ptr<ov::ICompiledModel> MockPlugin::import_model(std::istream& model,
                                                             const ov::RemoteContext& context,
                                                             const ov::AnyMap& properties) const {
    if (m_plugin)
        return m_plugin->import_model(model, context, properties);
    OPENVINO_NOT_IMPLEMENTED;
}
ov::SupportedOpsMap MockPlugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                            const ov::AnyMap& properties) const {
    if (m_plugin)
        return m_plugin->query_model(model, properties);
    OPENVINO_NOT_IMPLEMENTED;
}

struct Plugins {
    ov::IPlugin* plugin;
    InferenceEngine::IInferencePlugin* old_plugin;
};

std::queue<Plugins> targets;

OPENVINO_PLUGIN_API void CreatePluginEngine(std::shared_ptr<ov::IPlugin>& plugin) {
    if (targets.empty()) {
        ov::IPlugin* p = nullptr;
        plugin = std::make_shared<MockPlugin>(p);
    } else {
        auto plugins = targets.front();
        targets.pop();
        if (plugins.old_plugin == nullptr) {
            ov::IPlugin* p = nullptr;
            std::swap(plugins.plugin, p);
            plugin = std::make_shared<MockPlugin>(p);
        } else {
            InferenceEngine::IInferencePlugin* p = nullptr;
            std::swap(plugins.old_plugin, p);
            plugin = std::make_shared<MockPlugin>(p);
        }
    }
}

OPENVINO_PLUGIN_API void InjectProxyEngine(InferenceEngine::IInferencePlugin* target) {
    Plugins p;
    p.plugin = nullptr;
    p.old_plugin = target;
    targets.push(p);
}

OPENVINO_PLUGIN_API void InjectPlugin(ov::IPlugin* target) {
    Plugins p;
    p.plugin = target;
    p.old_plugin = nullptr;
    targets.push(p);
}
