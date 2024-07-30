// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_plugin.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/iplugin.hpp"

class MockInternalPlugin : public ov::IPlugin {
    ov::IPlugin* m_plugin = nullptr;
    ov::AnyMap config;

public:
    explicit MockInternalPlugin(ov::IPlugin* target) : m_plugin(target) {}
    explicit MockInternalPlugin() = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override {
        if (m_plugin)
            return m_plugin->compile_model(model, properties);
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                      const ov::AnyMap& properties) const override {
        if (m_plugin)
            return m_plugin->compile_model(model_path, properties);
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override {
        if (m_plugin)
            return m_plugin->compile_model(model, properties, context);
        OPENVINO_NOT_IMPLEMENTED;
    }

    void set_property(const ov::AnyMap& properties) override {
        config = properties;
        if (m_plugin) {
            m_plugin->set_property(config);
        }
    }

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
        if (m_plugin)
            return m_plugin->get_property(name, arguments);
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        if (m_plugin)
            return m_plugin->create_context(remote_properties);
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        if (m_plugin)
            return m_plugin->get_default_context(remote_properties);
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override {
        if (m_plugin)
            return m_plugin->import_model(model, properties);
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override {
        if (m_plugin)
            return m_plugin->import_model(model, context, properties);
        OPENVINO_NOT_IMPLEMENTED;
    }

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override {
        if (m_plugin)
            return m_plugin->query_model(model, properties);
        OPENVINO_NOT_IMPLEMENTED;
    }

    void set_parameters_if_need(const std::shared_ptr<ov::ICore>& core, const std::string& dev_name) const {
        if (m_plugin) {
            if (!m_plugin->get_core() && core) {
                m_plugin->set_core(core);
            }
            if (m_plugin->get_device_name().empty()) {
                m_plugin->set_device_name(dev_name);
            }
        }
    }
};

void MockPlugin::set_parameters_if_need() const {
    auto core = get_core();
    if (auto internal_plugin = std::dynamic_pointer_cast<const MockInternalPlugin>(m_plugin)) {
        internal_plugin->set_parameters_if_need(core, get_device_name());
    }
}

MockPlugin::MockPlugin(const std::shared_ptr<ov::IPlugin>& target) : m_plugin(target) {
    OPENVINO_ASSERT(m_plugin);
}

void MockPlugin::set_property(const ov::AnyMap& properties) {
    set_parameters_if_need();
    m_plugin->set_property(properties);
}

ov::Any MockPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    set_parameters_if_need();
    return m_plugin->get_property(name, arguments);
}

std::shared_ptr<ov::ICompiledModel> MockPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties) const {
    set_parameters_if_need();
    return m_plugin->compile_model(model, properties);
}

std::shared_ptr<ov::ICompiledModel> MockPlugin::compile_model(const std::string& model_path,
                                                              const ov::AnyMap& properties) const {
    set_parameters_if_need();
    return m_plugin->compile_model(model_path, properties);
}

std::shared_ptr<ov::ICompiledModel> MockPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties,
                                                              const ov::SoPtr<ov::IRemoteContext>& context) const {
    set_parameters_if_need();
    return m_plugin->compile_model(model, properties, context);
}

ov::SoPtr<ov::IRemoteContext> MockPlugin::create_context(const ov::AnyMap& remote_properties) const {
    set_parameters_if_need();
    return m_plugin->create_context(remote_properties);
}

ov::SoPtr<ov::IRemoteContext> MockPlugin::get_default_context(const ov::AnyMap& remote_properties) const {
    set_parameters_if_need();
    return m_plugin->get_default_context(remote_properties);
}

std::shared_ptr<ov::ICompiledModel> MockPlugin::import_model(std::istream& model, const ov::AnyMap& properties) const {
    set_parameters_if_need();
    return m_plugin->import_model(model, properties);
}
std::shared_ptr<ov::ICompiledModel> MockPlugin::import_model(std::istream& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties) const {
    set_parameters_if_need();
    return m_plugin->import_model(model, context, properties);
}
ov::SupportedOpsMap MockPlugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                            const ov::AnyMap& properties) const {
    set_parameters_if_need();
    return m_plugin->query_model(model, properties);
}

std::queue<std::shared_ptr<ov::IPlugin>> targets;
std::mutex targets_mutex;

OPENVINO_PLUGIN_API void create_plugin_engine(std::shared_ptr<ov::IPlugin>& plugin) {
    std::shared_ptr<ov::IPlugin> internal_plugin;
    if (targets.empty()) {
        internal_plugin = std::make_shared<MockInternalPlugin>();
    } else {
        std::lock_guard<std::mutex> lock(targets_mutex);
        internal_plugin = targets.front();
        targets.pop();
    }
    plugin = std::make_shared<MockPlugin>(internal_plugin);
}

OPENVINO_PLUGIN_API void InjectPlugin(ov::IPlugin* target) {
    std::lock_guard<std::mutex> lock(targets_mutex);
    targets.push(std::make_shared<MockInternalPlugin>(target));
}

OPENVINO_PLUGIN_API void ClearTargets() {
    std::lock_guard<std::mutex> lock(targets_mutex);
    targets = {};
}
