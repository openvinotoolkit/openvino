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

class TelemetryClient {
public:
    TelemetryClient();
    ~TelemetryClient();

    std::optional<float> utilization(const std::string& device_name, const std::string& device_type = "");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

inline constexpr std::string_view k_cpu_utilization_metric = "CPUUtilization";
inline constexpr std::string_view k_igpu_utilization_metric = "IGPUUtilization";
inline constexpr std::string_view k_dgpu_utilization_metric = "DGPUUtilization";
inline constexpr std::string_view k_npu_utilization_metric = "NPUUtilization";

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

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov
