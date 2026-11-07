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
 * @brief auto device setting that enables compiling the model to the remaining candidate devices in the background,
 * after the actual device is ready, to populate cache blobs for faster device switching.
 * Enabling it disables ov::intel_auto::enable_startup_fallback and ov::intel_auto::enable_runtime_fallback.
 * Requires ov::cache_dir or ov::cache_path, otherwise compiling the model throws on every host. Ignored in
 * cumulative throughput mode, when there is no other candidate device, and when
 * ov::cache_blob_id is set. Devices are skipped when the cache path is a single-file cache (*.bin).
 * In dynamic device selection, a device that is no longer selected releases its compiled model, which is loaded
 * from the cache when it is selected again; the initial device and devices without a cache location are kept.
 *
 * When ov::device::priorities (or a device string such as "AUTO:GPU,NPU") is set, it determines the candidates;
 * otherwise AUTO discovers its usual eligible devices. For example, to warm up the cache for GPU and NPU, so that
 * the one not selected by AUTO is compiled in the background:
 * @code
 * core.compile_model(model,
 *                    "AUTO",
 *                    ov::device::priorities("GPU,NPU"),
 *                    ov::cache_dir("model_cache"),
 *                    ov::intel_auto::compile_for_all(true));
 * @endcode
 */
static constexpr Property<bool> compile_for_all{"COMPILE_FOR_ALL"};

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
 * A nonempty value enables per-inference device selection for stateless, in-memory AUTO models with an alternative
 * device, unless the performance mode is THROUGHPUT or CUMULATIVE_THROUGHPUT, or device_bind_buffer is enabled.
 * In this mode the target device is selected for every inference and inference is serialized per compiled model.
 * Startup and runtime fallback are not supported: explicitly enabling ENABLE_STARTUP_FALLBACK or
 * ENABLE_RUNTIME_FALLBACK makes compilation fail. More than one stream is not supported, and MODEL_PRIORITY
 * is ignored. These restrictions apply only when per-inference selection is active.
 * The thresholds are ignored, and AUTO behaves as without this property, when the telemetry backend (IPF)
 * is not available.
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
 * Setting it enables the per inference device selection, which does not support startup or runtime fallback (see
 * devices_utilization_threshold).
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<PerfCurveTable> perf_curve_table{"PERF_CURVE_TABLE"};

/**
 * @brief Name of the device AUTO should prefer while the platform is in low power mode
 * (as reported by IPF/DTT). Takes precedence over
 * devices_utilization_threshold when the platform is in low power mode. The value is matched
 * against a candidate DeviceInformation::device_name first (e.g. "GPU.0"), then against its base
 * device name (e.g. "NPU" also matches a candidate named "NPU.5010").
 * On builds without OV_AUTO_ENABLE_IPF, or when the IPF backend cannot be initialized, the platform is never
 * considered to be in low power mode, so this property has no effect.
 * Otherwise setting it enables the per inference device selection, which does not support startup or runtime
 * fallback (see devices_utilization_threshold).
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string> low_power_device{"LOW_POWER_DEVICE"};
}  // namespace intel_auto
}  // namespace ov
