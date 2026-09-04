// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "../common.hpp"

namespace ov {
namespace auto_plugin {
namespace device_monitor {

// IPF (Intel(R) Innovation Platform Framework): Intel's telemetry/policy client API used
//     here to query platform state and subscribe to runtime platform events (e.g. DTT gear changes).
// DTT (Intel(R) Dynamic Tuning Technology): Intel driver/service that manages platform
//     power/thermal policy and exposes its state (status, version, current EPO gear) through IPF.
// EPO (Energy Performance Optimizer): one of DTT's policies; its "gear" (1-7) indicates whether
//     the platform currently favors low-latency/performance (gears 1-3) or low power (gears 4-7).

#ifdef OV_AUTO_ENABLE_IPF
void gear_changed_callback(const char* path, const char* event, void* context);
#endif

class TelemetryClient {
public:
    TelemetryClient();
    ~TelemetryClient();

    std::optional<float> utilization(const std::string& device_name, const std::string& device_type = "");

    // Whether the platform is currently in low power mode, based on startup CurrentGear state and
    // any later IPF/DTT OnEpoGearChanged notifications. std::nullopt means the mode is unknown.
    // Lazy-initializes DTT version/gear queries and event registration on first call.
    std::optional<bool> is_low_power_mode();

private:
#ifdef OV_AUTO_ENABLE_IPF
    friend void gear_changed_callback(const char* path, const char* event, void* context);
#endif
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

inline constexpr std::string_view k_cpu_utilization_metric = "CPUUtilization";
inline constexpr std::string_view k_igpu_utilization_metric = "IGPUUtilization";
// Fallback key: some platforms report integrated GPU utilization as "GPUUtilization".
inline constexpr std::string_view k_igpu_utilization_fallback_metric = "GPUUtilization";
inline constexpr std::string_view k_dgpu_utilization_metric = "DGPUUtilization";
inline constexpr std::string_view k_npu_utilization_metric = "NPUUtilization";
// EPO defines gears 1-7: gears 1-3 request low-latency/performance operation
// (perf_curve_table), gears 4-7 request low power operation (low_power_device).
inline constexpr int k_min_gear = 1;
inline constexpr int k_max_gear = 7;
inline constexpr int k_low_power_mode_min_gear = 4;
inline constexpr int k_low_power_mode_max_gear = k_max_gear;

inline constexpr bool is_valid_gear(int gear) {
    return gear >= k_min_gear && gear <= k_max_gear;
}

inline constexpr bool is_low_power_gear(int gear) {
    return gear >= k_low_power_mode_min_gear && gear <= k_low_power_mode_max_gear;
}

inline constexpr bool has_prefix(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

// CPU -> CPUUtilization; GPU(integrated/discrete) -> IGPU/DGPU; NPU -> NPUUtilization.
inline constexpr std::string_view device_to_metric_key(std::string_view device_name,
                                                       std::string_view device_type = {}) {
    std::string_view metric_key;
    if (has_prefix(device_name, "CPU")) {
        metric_key = k_cpu_utilization_metric;
    } else if (has_prefix(device_name, "GPU")) {
        if (device_type == "integrated") {
            metric_key = k_igpu_utilization_metric;
        } else if (device_type == "discrete") {
            metric_key = k_dgpu_utilization_metric;
        }
    } else if (has_prefix(device_name, "NPU")) {
        metric_key = k_npu_utilization_metric;
    }
    return metric_key;
}

#if defined(MULTIUNITTEST) && defined(OV_AUTO_ENABLE_IPF)
std::optional<float> parse_utilization_from_aiselector_json_for_test(const std::string& json_str,
                                                                     const std::string& device_name,
                                                                     const std::string& device_type = "");
#endif

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov
