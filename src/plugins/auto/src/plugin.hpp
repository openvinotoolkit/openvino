// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <filesystem>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "adaptive/signal_hub.hpp"
#include "cache_manager.hpp"
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
                  const std::optional<DeviceInformation>& low_power_device = {});
    MOCKTESTMACRO std::list<DeviceInformation> sort_device_by_perf_curve(
        const std::unordered_map<std::string, float>& device_utilizations,
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
    // Returns utilization values for the specified devices.
    MOCKTESTMACRO std::unordered_map<std::string, float> get_device_utilizations(
        const std::list<DeviceInformation>& devices);

    // Whether the platform is currently in low power mode; see device_monitor::TelemetryClient.
    MOCKTESTMACRO std::optional<bool> get_low_power_mode();

    /**
     * @brief Keep the utilization of the devices up to date in the background, so that selecting a device does not
     * query the telemetry backend.
     * @return the subscription, which ends with its last reference
     */
    // how the utilization was obtained so far, for the log
    std::string telemetry_summary();

    std::shared_ptr<SignalHub::Subscription> subscribe_telemetry(const std::vector<DeviceInformation>& devices,
                                                                 SignalHub::StatsFn stats,
                                                                 SignalHub::HeadroomFn headroom);

    // Whether the telemetry backend (IPF) is operational; without it the resource aware selection cannot work.
    MOCKTESTMACRO bool is_telemetry_available() const;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                             const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                             const ov::SoPtr<ov::IRemoteContext>& context,
                                                             const ov::AnyMap& properties) const override;

    CacheManager& get_cache_manager() {
        return *m_cache_manager;
    }

    const CacheManager& get_cache_manager() const {
        return *m_cache_manager;
    }

private:
    std::shared_ptr<ov::ICompiledModel> compile_model_impl(const std::filesystem::path& model_path,
                                                           const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties,
                                                           const std::string& model_precision = "FP32") const;
    // the config a device is compiled with: the supported user properties plus the default hints AUTO adds
    ov::AnyMap get_device_config(const DeviceName& device_with_id, const ov::AnyMap& properties) const;
    std::vector<DeviceInformation> filter_device(const std::vector<DeviceInformation>& meta_devices,
                                                 const ov::AnyMap& properties) const;
    std::vector<DeviceInformation> filter_device_by_model(const std::vector<DeviceInformation>& meta_devices,
                                                          const std::shared_ptr<const ov::Model>& model,
                                                          PluginConfig& load_config,
                                                          bool& is_stateful_model) const;
    std::string get_log_tag() const noexcept;
    // Base family name, perf_curve_table lookup key ("iGPU"/"dGPU" for GPUs via ov::device::type,
    // empty when it cannot be determined), and the ov::device::type string used to resolve a
    // device's IPF utilization metric key.
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
    mutable std::once_flag m_telemetry_client_init_once;
    mutable std::unique_ptr<device_monitor::TelemetryClient> m_telemetry_client;
    // the telemetry client takes one query at a time
    mutable std::mutex m_telemetry_mutex;
    std::atomic<size_t> m_sync_queries{0};
    std::atomic<size_t> m_hub_reads{0};
    // declared after the client: its thread stops first
    mutable std::mutex m_signal_hub_mutex;
    std::shared_ptr<SignalHub> m_signal_hub;
    std::unique_ptr<CacheManager> m_cache_manager = std::make_unique<CacheManager>();
};

}  // namespace auto_plugin
}  // namespace ov
