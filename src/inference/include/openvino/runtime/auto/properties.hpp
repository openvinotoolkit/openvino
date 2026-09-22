// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <openvino/runtime/properties.hpp>
#include <string>

namespace ov {

/**
 * @brief Namespace with Intel AUTO specific properties
 */
namespace intel_auto {
/**
 * @brief auto/multi device setting that enables performance improvement by binding buffer to hw infer request
 */
static constexpr Property<bool> device_bind_buffer{"DEVICE_BIND_BUFFER"};

/**
 * @brief auto device setting that enable/disable CPU as acceleration (or helper device) at the beginning
 */
static constexpr Property<bool> enable_startup_fallback{"ENABLE_STARTUP_FALLBACK"};

/**
 * @brief auto device setting that enable/disable runtime fallback to other devices when infer fails on current
 * selected device
 */
static constexpr Property<bool> enable_runtime_fallback{"ENABLE_RUNTIME_FALLBACK"};

/**
 * @brief Enum to define the policy of scheduling inference request to target device in cumulative throughput mode on
 * AUTO
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class SchedulePolicy {
    ROUND_ROBIN = 0,            // will schedule the infer request using round robin policy
    DEVICE_PRIORITY = 1,        // will schedule the infer request based on the device priority
    DEFAULT = DEVICE_PRIORITY,  //!<  Default schedule policy is DEVICE_PRIORITY
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SchedulePolicy& policy) {
    switch (policy) {
    case SchedulePolicy::ROUND_ROBIN:
        return os << "ROUND_ROBIN";
    case SchedulePolicy::DEVICE_PRIORITY:
        return os << "DEVICE_PRIORITY";
    default:
        OPENVINO_THROW("Unsupported schedule policy value");
    }
}

inline std::istream& operator>>(std::istream& is, SchedulePolicy& policy) {
    std::string str;
    is >> str;
    if (str == "ROUND_ROBIN") {
        policy = SchedulePolicy::ROUND_ROBIN;
    } else if (str == "DEVICE_PRIORITY") {
        policy = SchedulePolicy::DEVICE_PRIORITY;
    } else if (str == "DEFAULT") {
        policy = SchedulePolicy::DEFAULT;
    } else {
        OPENVINO_THROW("Unsupported schedule policy: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief High-level OpenVINO model policy hint
 * Defines what scheduling policy should be used in AUTO CUMULATIVE_THROUGHPUT or MULTI case
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<SchedulePolicy> schedule_policy{"SCHEDULE_POLICY"};

/**
 * @brief Device utilization thresholds (in percent) used by AUTO for device selection.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::map<std::string, unsigned>> devices_utilization_threshold{
    "DEVICES_UTILIZATION_THRESHOLD"};

/**
 * @brief Type of the ov::intel_auto::perf_curve_table property: maps a device key
 * ("CPU", "iGPU", "dGPU", "NPU") to a utilization-percent -> performance-score curve.
 * @ingroup ov_runtime_cpp_prop_api
 */
using PerfCurveTable = std::map<std::string, std::map<unsigned, float>>;

/**
 * @brief Per-device performance curve table mapping utilization percent to a relative performance score, used by
 * AUTO for device selection when set. A lower interpolated score indicates a more preferred device: AUTO ranks
 * candidates in ascending order of score and selects the one with the lowest score for the current utilization.
 * Device key must be one of "CPU", "iGPU", "dGPU", "NPU".
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<PerfCurveTable> perf_curve_table{"PERF_CURVE_TABLE"};

/**
 * @brief Name of the device AUTO should prefer while the platform is in low power mode
 * (as reported by IPF/DTT). Takes precedence over
 * devices_utilization_threshold when the platform is in low power mode. The value is matched
 * against a candidate DeviceInformation::device_name first (e.g. "GPU.0"), then against its base
 * device name (e.g. "NPU" also matches a candidate named "NPU.5010").
 * On builds without OV_AUTO_ENABLE_IPF, the telemetry backend always returns unknown, so this
 * property has no effect.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string> low_power_device{"LOW_POWER_DEVICE"};
}  // namespace intel_auto
}  // namespace ov
