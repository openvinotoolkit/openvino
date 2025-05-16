// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <string>
#include <list>

#include "openvino/runtime/iplugin.hpp"
#include "utils/log_util.hpp"
#include "common.hpp"
#include "plugin_config.hpp"
#include "compiled_model.hpp"

namespace ov {
namespace auto_plugin {

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin() = default;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties,
                                                              const ov::SoPtr<ov::IRemoteContext>& context) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                      const ov::AnyMap& properties) const override;

    MOCKTESTMACRO bool is_meta_device(const std::string& priorities) const;
    MOCKTESTMACRO std::vector<auto_plugin::DeviceInformation> parse_meta_devices(const std::string & devices_requests_cfg,
                                                                                 const ov::AnyMap& properties) const;

    MOCKTESTMACRO std::string get_device_list(const ov::AnyMap& properties) const;

    MOCKTESTMACRO std::list<DeviceInformation> get_valid_device(const std::vector<DeviceInformation>& meta_devices,
                                                   const std::string& model_precision = "FP32") const;

    MOCKTESTMACRO DeviceInformation select_device(const std::vector<DeviceInformation>& meta_devices,
                                                 const std::string& model_precision = "FP32",
                                                 unsigned int priority = 0);
    void unregister_priority(const unsigned int& priority, const std::string& device_name);
    void register_priority(const unsigned int& priority, const std::string& device_name);

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;


    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties) const override;

private:
    std::shared_ptr<ov::ICompiledModel> compile_model_impl(const std::string& model_path,
                                                           const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties,
                                                           const std::string& model_precision = "FP32") const;
    std::vector<DeviceInformation> filter_device(const std::vector<DeviceInformation>& meta_devices,
                                                 const ov::AnyMap& properties) const;
    std::vector<DeviceInformation> filter_device_by_model(const std::vector<DeviceInformation>& meta_devices,
                                                          const std::shared_ptr<const ov::Model>& model,
                                                          PluginConfig& load_config) const;
    std::string get_log_tag() const noexcept;
    static std::shared_ptr<std::mutex> m_mtx;
    static std::shared_ptr<std::map<unsigned int, std::list<std::string>>> m_priority_map;
    PluginConfig m_plugin_config;
};

}  // namespace auto_plugin
}  // namespace ov
