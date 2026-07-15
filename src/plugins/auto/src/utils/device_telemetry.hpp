// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <string>

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define auto_plugin mock_auto_plugin
#else
#define MOCKTESTMACRO
#endif

namespace ov {
namespace auto_plugin {
namespace device_monitor {

class TelemetryClient {
public:
    TelemetryClient();
    ~TelemetryClient();

    std::optional<float> utilization(const std::string& device_name, const std::string& device_luid = "");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

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

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov
