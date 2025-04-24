// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <memory>

#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

#define OV_PLUGIN_CALL_STATEMENT(...)                                                  \
    OPENVINO_ASSERT(m_ptr != nullptr, "OpenVINO Runtime Plugin was not initialized."); \
    try {                                                                              \
        __VA_ARGS__;                                                                   \
    } catch (const std::exception& ex) {                                               \
        OPENVINO_THROW(ex.what());                                                     \
    } catch (...) {                                                                    \
        OPENVINO_THROW("Unexpected exception");                                        \
    }

ov::Plugin::~Plugin() {
    m_ptr = {};
}

ov::Plugin::Plugin(const std::shared_ptr<ov::IPlugin>& ptr, const std::shared_ptr<void>& so) : m_ptr{ptr}, m_so{so} {
    OV_PLUGIN_CALL_STATEMENT();
}

void ov::Plugin::set_name(const std::string& deviceName) {
    OV_PLUGIN_CALL_STATEMENT({ m_ptr->set_device_name(deviceName); });
}

const std::string& ov::Plugin::get_name() const {
    OV_PLUGIN_CALL_STATEMENT({ return m_ptr->get_device_name(); });
}

void ov::Plugin::set_core(std::weak_ptr<ICore> core) {
    OV_PLUGIN_CALL_STATEMENT({ m_ptr->set_core(core); });
}

const ov::Version ov::Plugin::get_version() const {
    OV_PLUGIN_CALL_STATEMENT(return m_ptr->get_version());
}

void ov::Plugin::set_property(const ov::AnyMap& config) {
    m_ptr->set_property(config);
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

ov::SoPtr<ov::ICompiledModel> ov::Plugin::import_model(std::istream& model,
                                                       const ov::SoPtr<ov::IRemoteContext>& context,
                                                       const ov::AnyMap& config) const {
    OV_PLUGIN_CALL_STATEMENT(return {m_ptr->import_model(model, context, config), m_so});
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
    return {m_ptr->get_property(name, arguments), {m_so}};
}

bool ov::Plugin::supports_model_caching(const ov::AnyMap& arguments) const {
    bool supported(false);
    supported =
        util::contains(get_property(ov::supported_properties), ov::device::capabilities) &&
        util::contains(get_property(ov::device::capabilities, arguments), ov::device::capability::EXPORT_IMPORT) &&
        util::contains(get_property(ov::internal::supported_properties), ov::internal::caching_properties);
    return supported;
}
