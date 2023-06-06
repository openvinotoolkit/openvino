// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace proxy {

/**
 * @brief Creates a new instance of Proxy plugin
 *
 * @param plugin shared pointer to the plugin
 */
void create_plugin(std::shared_ptr<ov::IPlugin>& plugin);

/**
 * @brief Get wrapped remote context
 *
 * @param context Remote context
 *
 * @return Original remote context
 */
const std::shared_ptr<ov::IRemoteContext>& get_hardware_context(const std::shared_ptr<ov::IRemoteContext>& context);

/**
 * @brief Get wrapped remote tensor
 *
 * @param tensor Remote tensor
 *
 * @return Original remote tensor
 */
const std::shared_ptr<ov::ITensor>& get_hardware_tensor(const std::shared_ptr<ov::ITensor>& tensor);

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin();

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::RemoteContext& context) const override;

    std::shared_ptr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::RemoteContext& context,
                                                     const ov::AnyMap& properties) const override;

private:
    std::vector<std::vector<std::string>> get_hidden_devices() const;
    std::string get_fallback_device(size_t idx) const;
    std::string get_primary_device(size_t idx) const;

    ov::Any get_internal_property(const std::string& property_name, const std::string& conf_name = "") const;
    bool has_internal_property(const std::string& property_name, const std::string& conf_name = "") const;
    size_t get_device_from_config(const ov::AnyMap& config) const;

    size_t m_default_device = 0;
    std::vector<std::string> m_device_order;
    std::unordered_set<std::string> m_alias_for;
    // Update per device config in get_hidden_devices
    mutable std::unordered_map<std::string, ov::AnyMap> m_configs;
    mutable std::mutex m_plugin_mutex;
};

}  // namespace proxy
}  // namespace ov
