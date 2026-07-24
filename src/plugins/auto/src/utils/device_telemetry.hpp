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

    std::optional<float> utilization(const std::string& device_name, const std::string& device_type = "");

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Map an OpenVINO device to the platform telemetry metric key.
 *
 * "CPU"                     -> "CPUUtilization"
 * "GPU" + "integrated" type -> "IGPUUtilization"
 * "GPU" + "discrete" type   -> "DGPUUtilization"
 * "NPU"                     -> "NPUUtilization"
 *
 * For a GPU device an empty or unrecognized device_type yields an empty key so the
 * caller treats utilization as unavailable and keeps the device as a candidate.
 *
 * @param device_name OpenVINO device identifier (e.g. "CPU", "GPU", "GPU.0", "NPU").
 * @param device_type ov::device::type value ("integrated" or "discrete"); only used for GPU.
 * @return Telemetry metric key, or an empty string for an unknown device type.
 */
inline std::string device_to_metric_key(const std::string& device_name, const std::string& device_type = "") {
    if (device_name.rfind("CPU", 0) == 0) {
        return "CPUUtilization";
    }
    if (device_name.rfind("GPU", 0) == 0) {
        if (device_type == "integrated") {
            return "IGPUUtilization";
        }
        if (device_type == "discrete") {
            return "DGPUUtilization";
        }
        return "";
    }
    if (device_name.rfind("NPU", 0) == 0) {
        return "NPUUtilization";
    }
    return "";
}

}  // namespace device_monitor
}  // namespace auto_plugin
}  // namespace ov
