// Copyright (C) 2018-2023 Intel Corporation
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

    MOCKTESTMACRO std::vector<auto_plugin::DeviceInformation> parse_meta_devices(const std::string & devices_requests_cfg,
                                                                                 const ov::AnyMap& properties) const;

    MOCKTESTMACRO std::string get_device_list(const ov::AnyMap& properties) const;

    MOCKTESTMACRO std::list<DeviceInformation> get_valid_device(const std::vector<DeviceInformation>& meta_devices,
                                                   const std::string& network_precision = METRIC_VALUE(FP32));

    MOCKTESTMACRO DeviceInformation select_device(const std::vector<DeviceInformation>& meta_devices,
                                                 const std::string& network_precision = METRIC_VALUE(FP32),
                                                 unsigned int priority = 0);
    void unregister_priority(const unsigned int& priority, const std::string& device_name);
    void register_priority(const unsigned int& priority, const std::string& device_name);

protected:
    ov::AnyMap pre_process_config(const std::map<std::string, std::string>& orig_config) const;

private:
    std::shared_ptr<ov::ICompiledModel> compile_model_impl(const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties,
                                                           const std::string& networkPrecision = METRIC_VALUE(FP32));
    std::vector<DeviceInformation> filter_device(const std::vector<DeviceInformation>& meta_devices,
                                                const std::map<std::string, std::string>& config);
    std::vector<DeviceInformation> filter_device_by_network(const std::vector<DeviceInformation>& meta_devices,
                                                         InferenceEngine::CNNNetwork network);
    std::string get_log_tag() const noexcept;
    static std::mutex m_mtx;
    static std::map<unsigned int, std::list<std::string>> m_priority_map;
    std::string m_log_tag;
    PluginConfig m_plugin_config;
};

}  // namespace auto_plugin
}  // namespace ov
