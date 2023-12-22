// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <memory>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_plugin_config.hpp"
#include "iplugin_wrapper.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

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

void ov::Plugin::add_extension(const InferenceEngine::IExtensionPtr& extension) {
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
                                                        const ov::SoPtr<ov::IRemoteContext>& context,
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
                                                       const ov::SoPtr<ov::IRemoteContext>& context,
                                                       const ov::AnyMap& config) const {
    OV_PLUGIN_CALL_STATEMENT(return {m_ptr->import_model(networkModel, context, config), m_so});
}

ov::SoPtr<ov::IRemoteContext> ov::Plugin::create_context(const AnyMap& params) const {
    OV_PLUGIN_CALL_STATEMENT({
        auto remote = m_ptr->create_context(params);
        if (!remote._so)
            remote._so = m_so;
        return remote;
    });
}

ov::SoPtr<ov::IRemoteContext> ov::Plugin::get_default_context(const AnyMap& params) const {
    OV_PLUGIN_CALL_STATEMENT({
        auto remote = m_ptr->get_default_context(params);
        if (!remote._so)
            remote._so = m_so;
        return remote;
    });
}

ov::Any ov::Plugin::get_property(const std::string& name, const AnyMap& arguments) const {
    OV_PLUGIN_CALL_STATEMENT({
        if (ov::supported_properties == name) {
            try {
                return {m_ptr->get_property(name, arguments), {m_so}};
            } catch (const InferenceEngine::Exception&) {
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
                } catch (const InferenceEngine::Exception&) {
                }
                try {
                    auto rw_properties = m_ptr->get_property(METRIC_KEY(SUPPORTED_CONFIG_KEYS), arguments)
                                             .as<std::vector<std::string>>();
                    for (auto&& rw_property : rw_properties) {
                        supported_properties.emplace_back(rw_property, PropertyMutability::RW);
                    }
                } catch (const ov::Exception&) {
                } catch (const InferenceEngine::Exception&) {
                }
                supported_properties.emplace_back(ov::supported_properties.name(), PropertyMutability::RO);
                return supported_properties;
            }
        }
        // Add legacy supported properties
        if (METRIC_KEY(SUPPORTED_METRICS) == name || METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
            try {
                return {m_ptr->get_property(name, arguments), {m_so}};
            } catch (const ov::Exception&) {
                auto props =
                    m_ptr->get_property(ov::supported_properties.name(), arguments).as<std::vector<PropertyName>>();
                std::vector<std::string> legacy_properties;
                for (const auto& prop : props) {
                    if ((METRIC_KEY(SUPPORTED_METRICS) == name && !prop.is_mutable()) ||
                        (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name && prop.is_mutable()))
                        legacy_properties.emplace_back(prop);
                }
                if (METRIC_KEY(SUPPORTED_METRICS) == name) {
                    legacy_properties.emplace_back(METRIC_KEY(SUPPORTED_METRICS));
                    legacy_properties.emplace_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                }
                if (METRIC_KEY(SUPPORTED_METRICS) == name && supports_model_caching(false))
                    legacy_properties.emplace_back(METRIC_KEY(IMPORT_EXPORT_SUPPORT));

                return legacy_properties;
            }
        }
        if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
            try {
                return {m_ptr->get_property(name, arguments), {m_so}};
            } catch (const ov::Exception&) {
                if (!supports_model_caching(false))
                    throw;
                // if device has ov::device::capability::EXPORT_IMPORT it means always true
                return true;
            }
        }
        return {m_ptr->get_property(name, arguments), {m_so}};
    });
}

bool ov::Plugin::supports_model_caching(bool check_old_api) const {
    bool supported(false);
    if (check_old_api) {
        auto supportedMetricKeys = get_property(METRIC_KEY(SUPPORTED_METRICS), {}).as<std::vector<std::string>>();
        supported = util::contains(supportedMetricKeys, METRIC_KEY(IMPORT_EXPORT_SUPPORT)) &&
                    get_property(METRIC_KEY(IMPORT_EXPORT_SUPPORT), {}).as<bool>();
    }
    if (!supported) {
        supported = util::contains(get_property(ov::supported_properties), ov::device::capabilities) &&
                    util::contains(get_property(ov::device::capabilities), ov::device::capability::EXPORT_IMPORT);
    }
    if (supported) {
        supported = util::contains(get_property(ov::internal::supported_properties), ov::internal::caching_properties);
    }
    return supported;
}
