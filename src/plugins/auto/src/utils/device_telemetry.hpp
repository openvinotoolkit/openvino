// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>

namespace ov {
namespace auto_plugin {
namespace device_monitor {

/**
 * @brief Map an OpenVINO device name to the platform telemetry metric key.
 *
 * "CPU"           -> "CPUUtilization"
 * "GPU", "GPU.0"  -> "GPUUtilization"
 * "NPU"           -> "NPUUtilization"
 *
 * @param device_name OpenVINO device identifier (e.g. "CPU", "GPU", "GPU.0", "NPU").
 * @return Telemetry metric key, or an empty string for an unknown device type.
 */
inline std::string device_name_to_metric_key(const std::string& device_name) {
    if (device_name.rfind("CPU", 0) == 0) {
        return "CPUUtilization";
    }
    if (device_name.rfind("GPU", 0) == 0) {
        return "GPUUtilization";
    }
    if (device_name.rfind("NPU", 0) == 0) {
        return "NPUUtilization";
    }
    return "";
}

/**
 * @brief Query the utilization of a device from platform telemetry.
 *
 * The utilization source is provided by the platform telemetry backend. When the
 * backend is unavailable (unsupported platform, backend not initialized, telemetry
 * missing, or a parsing error), std::nullopt is returned so callers can gracefully
 * skip the utilization-based filtering.
 *
 * @param device_name OpenVINO device identifier (e.g. "CPU", "GPU", "GPU.0", "NPU").
 * @param device_luid Optional platform-specific device identifier (reserved).
 * @return Utilization in percent within [0.0, 100.0], or std::nullopt if unavailable.
 */
std::optional<float> query_device_utilization(const std::string& device_name, const std::string& device_luid = "");

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov
