// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <memory>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_plugin_config.hpp"
#include "iplugin_wrapper.hpp"

#define OV_PLUGIN_CALL_STATEMENT(...)                                                  \
    OPENVINO_ASSERT(m_ptr != nullptr, "OpenVINO Runtime Plugin was not initialized."); \
    try {                                                                              \
        __VA_ARGS__;                                                                   \
    } catch (...) {                                                                    \
        ::InferenceEngine::details::Rethrow();                                         \
    }
ov::Plugin::~Plugin() {
    m_ptr = {};
}

ov::Plugin::Plugin(const std::shared_ptr<ov::IPlugin>& ptr, const std::shared_ptr<void>& so) : m_ptr{ptr}, m_so{so} {
    OV_PLUGIN_CALL_STATEMENT();
}

void ov::Plugin::set_name(const std::string& deviceName) {
    OV_PLUGIN_CALL_STATEMENT({
        m_ptr->set_device_name(deviceName);
        if (auto wrapper = std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(m_ptr))
            wrapper->set_device_name(deviceName);
    });
}

const std::string& ov::Plugin::get_name() const {
    OV_PLUGIN_CALL_STATEMENT({ return m_ptr->get_device_name(); });
}

void ov::Plugin::set_core(std::weak_ptr<ICore> core) {
    OV_PLUGIN_CALL_STATEMENT({
        m_ptr->set_core(core);
        if (auto wrapper = std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(m_ptr))
            wrapper->set_core(core);
    });
}

const ov::Version ov::Plugin::get_version() const {
    OV_PLUGIN_CALL_STATEMENT(return m_ptr->get_version());
}

void ov::Plugin::add_extension(const ie::IExtensionPtr& extension) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    OV_PLUGIN_CALL_STATEMENT(m_ptr->add_extension(extension));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void ov::Plugin::set_property(const ov::AnyMap& config) {
    OV_PLUGIN_CALL_STATEMENT(m_ptr->set_property(config));
}

ov::SoPtr<ov::ICompiledModel> ov::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                        const ov::AnyMap& properties) const {
    OV_PLUGIN_CALL_STATEMENT(return {m_ptr->compile_model(model, properties), m_so});
}

ov::SoPtr<ov::ICompiledModel> ov::Plugin::compile_model(const std::string& model_path,
                                                        const ov::AnyMap& properties) const {
    OV_PLUGIN_CALL_STATEMENT(return {m_ptr->compile_model(model_path, properties), m_so});
}

ov::SoPtr<ov::ICompiledModel> ov::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                        const ov::RemoteContext& context,
                                                        const ov::AnyMap& properties) const {
    OV_PLUGIN_CALL_STATEMENT(return {m_ptr->compile_model(model, properties, context), m_so});
}

ov::SupportedOpsMap ov::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                            const ov::AnyMap& properties) const {
    OV_PLUGIN_CALL_STATEMENT(return m_ptr->query_model(model, properties));
}

ov::SoPtr<ov::ICompiledModel> ov::Plugin::import_model(std::istream& model, const ov::AnyMap& properties) const {
    OV_PLUGIN_CALL_STATEMENT(return {m_ptr->import_model(model, properties), m_so});
}

ov::SoPtr<ov::ICompiledModel> ov::Plugin::import_model(std::istream& networkModel,
                                                       const ov::RemoteContext& context,
                                                       const ov::AnyMap& config) const {
    OV_PLUGIN_CALL_STATEMENT(return {m_ptr->import_model(networkModel, context, config), m_so});
}

ov::RemoteContext ov::Plugin::create_context(const AnyMap& params) const {
    OV_PLUGIN_CALL_STATEMENT({
        auto remote = m_ptr->create_context(params);
        auto so = remote._so;
        if (m_so)
            so.emplace_back(m_so);
        return {remote._impl, so};
    });
}

ov::RemoteContext ov::Plugin::get_default_context(const AnyMap& params) const {
    OV_PLUGIN_CALL_STATEMENT({
        auto remote = m_ptr->get_default_context(params);
        auto so = remote._so;
        if (m_so)
            so.emplace_back(m_so);
        return {remote._impl, so};
    });
}

ov::Any ov::Plugin::get_property(const std::string& name, const AnyMap& arguments) const {
    OV_PLUGIN_CALL_STATEMENT({
        if (ov::supported_properties == name) {
            try {
                return {m_ptr->get_property(name, arguments), {m_so}};
            } catch (const ie::Exception&) {
                std::vector<ov::PropertyName> supported_properties;
                try {
                    auto ro_properties =
                        m_ptr->get_property(METRIC_KEY(SUPPORTED_METRICS), arguments).as<std::vector<std::string>>();
                    for (auto&& ro_property : ro_properties) {
                        if (ro_property != METRIC_KEY(SUPPORTED_METRICS) &&
                            ro_property != METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                            supported_properties.emplace_back(ro_property, PropertyMutability::RO);
                        }
                    }
                } catch (const ov::Exception&) {
                } catch (const ie::Exception&) {
                }
                try {
                    auto rw_properties = m_ptr->get_property(METRIC_KEY(SUPPORTED_CONFIG_KEYS), arguments)
                                             .as<std::vector<std::string>>();
                    for (auto&& rw_property : rw_properties) {
                        supported_properties.emplace_back(rw_property, PropertyMutability::RW);
                    }
                } catch (const ov::Exception&) {
                } catch (const ie::Exception&) {
                }
                supported_properties.emplace_back(ov::supported_properties.name(), PropertyMutability::RO);
                return supported_properties;
            }
        }
        return {m_ptr->get_property(name, arguments), {m_so}};
    });
}
