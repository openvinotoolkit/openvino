// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <filesystem>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"
#include "compiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "plugin_config.hpp"
#include "utils/device_telemetry.hpp"
#include "utils/log_util.hpp"

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

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::filesystem::path& model_path,
                                                      const ov::AnyMap& properties) const override;

    MOCKTESTMACRO bool is_meta_device(const std::string& priorities) const;
    MOCKTESTMACRO std::vector<auto_plugin::DeviceInformation> parse_meta_devices(const std::string & devices_requests_cfg,
                                                                                 const ov::AnyMap& properties) const;

    MOCKTESTMACRO std::string get_device_list(ov::AnyMap& properties,
                                              const std::shared_ptr<const ov::Model>& model = nullptr,
                                              const std::filesystem::path& model_path = {}) const;

    MOCKTESTMACRO std::list<DeviceInformation> get_valid_device(const std::vector<DeviceInformation>& meta_devices,
                                                                const std::string& model_precision = "FP32") const;

    MOCKTESTMACRO DeviceInformation
    select_device(const std::vector<DeviceInformation>& meta_devices,
                  const std::string& model_precision = "FP32",
                  unsigned int priority = 0,
                  const DeviceSelectionPolicy& selection_policy = {},
                  const std::string& low_power_device = {});
    MOCKTESTMACRO std::list<DeviceInformation> sort_device_by_perf_curve(
        const std::list<DeviceInformation>& valid_devices,
        const ov::intel_auto::PerfCurveTable& perf_curve_table,
        size_t* out_scored_count = nullptr);
    void unregister_priority(const unsigned int& priority, const std::string& device_name);
    void register_priority(const unsigned int& priority, const std::string& device_name);

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;


    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties) const override;
    MOCKTESTMACRO std::optional<float> get_device_utilization(const std::string& device_name,
                                                              const std::string& device_type = "");

    // Whether the platform is currently in low power mode; see device_monitor::TelemetryClient.
    MOCKTESTMACRO std::optional<bool> get_low_power_mode();

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                             const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties) const override;

private:
    std::shared_ptr<ov::ICompiledModel> compile_model_impl(const std::filesystem::path& model_path,
                                                           const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties,
                                                           const std::string& model_precision = "FP32") const;
    std::vector<DeviceInformation> filter_device(const std::vector<DeviceInformation>& meta_devices,
                                                 const ov::AnyMap& properties) const;
    std::vector<DeviceInformation> filter_device_by_model(const std::vector<DeviceInformation>& meta_devices,
                                                          const std::shared_ptr<const ov::Model>& model,
                                                          PluginConfig& load_config) const;
    std::string get_log_tag() const noexcept;
    // Base family name, perf_curve_table lookup key ("iGPU"/"dGPU" for GPUs via ov::device::type,
    // empty when it cannot be determined), and the ov::device::type string used by get_device_utilization.
    struct DeviceKey {
        std::string base_name;
        std::string logical_key;
        std::string device_type;
    };
    DeviceKey resolve_device_key(const std::string& device_name) const;
    static float interpolate_perf_score(const std::map<unsigned, float>& curve, float utilization);
    static std::shared_ptr<std::mutex> m_mtx;
    static std::shared_ptr<std::map<unsigned int, std::list<std::string>>> m_priority_map;
    PluginConfig m_plugin_config;
    std::once_flag m_telemetry_client_init_once;
    std::unique_ptr<device_monitor::TelemetryClient> m_telemetry_client;
};

}  // namespace auto_plugin
}  // namespace ov
