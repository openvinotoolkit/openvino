// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/legacy_api_helper.hpp"

namespace ov {
namespace intel_gpu {

bool LegacyAPIHelper::is_new_api_property(const std::pair<std::string, ov::Any>& property) {
    static const std::vector<std::string> new_properties_list = {
        ov::intel_gpu::hint::queue_priority.name(),
        ov::intel_gpu::hint::queue_throttle.name(),
        ov::hint::inference_precision.name(),
        ov::compilation_num_threads.name(),
        ov::num_streams.name(),
    };

    return std::find(new_properties_list.begin(), new_properties_list.end(), property.first) != new_properties_list.end();
}

bool LegacyAPIHelper::is_legacy_property(const std::pair<std::string, ov::Any>& property, bool is_new_api) {
    static const std::vector<std::string> legacy_properties_list = {
        InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS,
        InferenceEngine::GPUConfigParams::KEY_GPU_MAX_NUM_THREADS,
        InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY,
        InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE,
    };

    static const std::vector<std::string> legacy_property_values_list = {
        InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
        InferenceEngine::GPUConfigParams::KEY_GPU_HOST_TASK_PRIORITY,
    };

    bool legacy_property = std::find(legacy_properties_list.begin(), legacy_properties_list.end(), property.first) != legacy_properties_list.end();
    bool need_value_conversion = !is_new_api &&
        std::find(legacy_property_values_list.begin(), legacy_property_values_list.end(), property.first) != legacy_property_values_list.end();

    return legacy_property || need_value_conversion;
}

ov::AnyMap LegacyAPIHelper::convert_legacy_properties(const std::map<std::string, std::string>& properties, bool is_new_api) {
    return convert_legacy_properties(ov::AnyMap(properties.begin(), properties.end()), is_new_api);
}

ov::AnyMap LegacyAPIHelper::convert_legacy_properties(const ov::AnyMap& properties, bool is_new_api) {
    ov::AnyMap converted_properties;
    for (auto& property : properties) {
        if (is_legacy_property(property, is_new_api)) {
            auto new_property = convert_legacy_property(property);
            converted_properties[new_property.first] = new_property.second;
        } else {
            converted_properties[property.first] = property.second;
        }
    }

    return converted_properties;
}

std::pair<std::string, ov::Any> LegacyAPIHelper::convert_legacy_property(const std::pair<std::string, ov::Any>& legacy_property) {
    auto legacy_name = legacy_property.first;
    if (legacy_name == InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) {
        ov::Any converted_val{legacy_property.second};
        auto legacy_val = legacy_property.second.as<std::string>();
        if (legacy_val == InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO)
            converted_val = ov::streams::AUTO;

        return { ov::num_streams.name(), converted_val };
    } else if (legacy_name == InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY) {
        ov::Any converted_val{nullptr};
        auto legacy_val = legacy_property.second.as<std::string>();
        if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH) {
            converted_val = ov::hint::Priority::HIGH;
        } else if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED) {
            converted_val = ov::hint::Priority::MEDIUM;
        } else if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW) {
            converted_val = ov::hint::Priority::LOW;
        } else {
            converted_val = legacy_val;
        }

        return { ov::hint::model_priority.name(), converted_val };
    } else if (legacy_name == InferenceEngine::GPUConfigParams::KEY_GPU_MAX_NUM_THREADS) {
        return { ov::compilation_num_threads.name(), legacy_property.second };
    } else if (legacy_name == InferenceEngine::GPUConfigParams::KEY_GPU_HOST_TASK_PRIORITY) {
        ov::Any converted_val{nullptr};
        auto legacy_val = legacy_property.second.as<std::string>();
        if (legacy_val == InferenceEngine::GPUConfigParams::GPU_HOST_TASK_PRIORITY_HIGH) {
            converted_val = ov::hint::Priority::HIGH;
        } else if (legacy_val == InferenceEngine::GPUConfigParams::GPU_HOST_TASK_PRIORITY_MEDIUM) {
            converted_val = ov::hint::Priority::MEDIUM;
        } else if (legacy_val == InferenceEngine::GPUConfigParams::GPU_HOST_TASK_PRIORITY_LOW) {
            converted_val = ov::hint::Priority::LOW;
        } else {
            converted_val = legacy_val;
        }
        return { ov::intel_gpu::hint::host_task_priority.name(), converted_val };
    } else if (legacy_name == InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY) {
        ov::Any converted_val{nullptr};
        auto legacy_val = legacy_property.second.as<std::string>();
        if (!legacy_val.empty()) {
            std::stringstream ss(legacy_val);
            uint32_t uVal(0);
            ss >> uVal;
            OPENVINO_ASSERT(!ss.fail(), "[GPU] Unsupported property value by plugin: ", legacy_val);
            switch (uVal) {
            case 0:
            case 2:
                converted_val = ov::hint::Priority::MEDIUM;
                break;
            case 1:
                converted_val = ov::hint::Priority::LOW;
                break;
            case 3:
                converted_val = ov::hint::Priority::HIGH;
                break;
            default:
                OPENVINO_ASSERT(false, "[GPU] Unsupported queue priority value ", uVal);
            }
        }

        return { ov::intel_gpu::hint::queue_priority.name(), converted_val };
    } else if (legacy_name == InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE) {
        ov::Any converted_val{nullptr};
        auto legacy_val = legacy_property.second.as<std::string>();
        if (!legacy_val.empty()) {
            std::stringstream ss(legacy_val);
            uint32_t uVal(0);
            ss >> uVal;
            OPENVINO_ASSERT(!ss.fail(), "[GPU] Unsupported property value by plugin: ", legacy_val);
            switch (uVal) {
            case 0:
            case 2:
                converted_val = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
                break;
            case 1:
                converted_val = ov::intel_gpu::hint::ThrottleLevel::LOW;
                break;
            case 3:
                converted_val = ov::intel_gpu::hint::ThrottleLevel::HIGH;
                break;
            default:
                OPENVINO_ASSERT(false, "[GPU] Unsupported queue throttle value ", uVal);
            }
        }

        return { ov::intel_gpu::hint::queue_throttle.name(), converted_val };
    }

    OPENVINO_ASSERT(false, "[GPU] Unhandled legacy property in convert_legacy_property method: ", legacy_property.first);
}

std::pair<std::string, ov::Any> LegacyAPIHelper::convert_to_legacy_property(const std::pair<std::string, ov::Any>& property) {
    auto name = property.first;
    if (name == ov::num_streams.name()) {
        ov::Any legacy_val{property.second};
        if (!property.second.empty()) {
            if (property.second.as<ov::streams::Num>() == ov::streams::AUTO) {
                legacy_val = InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO;
            }
        }

        return { InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, legacy_val };
    } else if (name == ov::hint::model_priority.name()) {
        ov::Any legacy_val{nullptr};
        if (!property.second.empty()) {
            ov::hint::Priority val = property.second.as<ov::hint::Priority>();
            switch (val) {
            case ov::hint::Priority::LOW: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW; break;
            case ov::hint::Priority::MEDIUM: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED; break;
            case ov::hint::Priority::HIGH: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH; break;
            default: OPENVINO_ASSERT(false, "[GPU] Unsupported model priority value ", val);
            }
        }

        return { InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, legacy_val };
    } else if (name == ov::compilation_num_threads.name()) {
        return { InferenceEngine::GPUConfigParams::KEY_GPU_MAX_NUM_THREADS, property.second };
    } else if (name == ov::intel_gpu::hint::host_task_priority.name()) {
        ov::Any legacy_val{nullptr};
        if (!property.second.empty()) {
            ov::hint::Priority val = property.second.as<ov::hint::Priority>();
            switch (val) {
            case ov::hint::Priority::LOW: legacy_val = InferenceEngine::GPUConfigParams::GPU_HOST_TASK_PRIORITY_LOW; break;
            case ov::hint::Priority::MEDIUM: legacy_val = InferenceEngine::GPUConfigParams::GPU_HOST_TASK_PRIORITY_MEDIUM; break;
            case ov::hint::Priority::HIGH: legacy_val = InferenceEngine::GPUConfigParams::GPU_HOST_TASK_PRIORITY_HIGH; break;
            default: OPENVINO_ASSERT(false, "[GPU] Unsupported host task priority value ", val);
            }
        }

        return { InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, legacy_val };
    } else if (name == ov::intel_gpu::hint::queue_priority.name()) {
        ov::Any legacy_val{nullptr};
        if (!property.second.empty()) {
            ov::hint::Priority val = property.second.as<ov::hint::Priority>();
            switch (val) {
            case ov::hint::Priority::LOW: legacy_val = "1"; break;
            case ov::hint::Priority::MEDIUM: legacy_val = "2"; break;
            case ov::hint::Priority::HIGH: legacy_val = "3"; break;
            default: OPENVINO_ASSERT(false, "[GPU] Unsupported queue throttle value ", val);
            }
        }

        return { InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY, legacy_val };
    } else if (name == ov::intel_gpu::hint::queue_throttle.name()) {
        ov::Any legacy_val{nullptr};
        if (!property.second.empty()) {
            ov::intel_gpu::hint::ThrottleLevel val = property.second.as<ov::intel_gpu::hint::ThrottleLevel>();
            switch (val) {
            case ov::intel_gpu::hint::ThrottleLevel::LOW: legacy_val = "1"; break;
            case ov::intel_gpu::hint::ThrottleLevel::MEDIUM: legacy_val = "2"; break;
            case ov::intel_gpu::hint::ThrottleLevel::HIGH: legacy_val = "3"; break;
            default: OPENVINO_ASSERT(false, "[GPU] Unsupported queue throttle value ", val);
            }
        }
        return { InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE, legacy_val };
    }

    OPENVINO_ASSERT(false, "[GPU] Unhandled legacy property in convert_to_legacy_property method: ", property.first);
}

std::vector<std::string> LegacyAPIHelper::get_supported_configs() {
    OPENVINO_SUPPRESS_DEPRECATED_START
    static const std::vector<std::string> supported_config = {
        CONFIG_KEY(MODEL_PRIORITY),
        CONFIG_KEY(PERFORMANCE_HINT),
        CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS),
        CONFIG_KEY(PERF_COUNT),
        CONFIG_KEY(CONFIG_FILE),
        CONFIG_KEY(DEVICE_ID),
        CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
        CONFIG_KEY(CACHE_DIR),
        CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
        GPU_CONFIG_KEY(PLUGIN_PRIORITY),
        GPU_CONFIG_KEY(PLUGIN_THROTTLE),
        GPU_CONFIG_KEY(HOST_TASK_PRIORITY),
        GPU_CONFIG_KEY(NV12_TWO_INPUTS),
        GPU_CONFIG_KEY(MAX_NUM_THREADS),
        GPU_CONFIG_KEY(ENABLE_LOOP_UNROLLING),
    };
    OPENVINO_SUPPRESS_DEPRECATED_END

    return supported_config;
}

std::vector<std::string> LegacyAPIHelper::get_supported_metrics() {
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::vector<std::string> supported_metrics = {
        METRIC_KEY(AVAILABLE_DEVICES),
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(FULL_DEVICE_NAME),
        METRIC_KEY(OPTIMIZATION_CAPABILITIES),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
        METRIC_KEY(RANGE_FOR_STREAMS),
        METRIC_KEY(DEVICE_TYPE),
        METRIC_KEY(DEVICE_GOPS),
        METRIC_KEY(OPTIMAL_BATCH_SIZE),
        METRIC_KEY(MAX_BATCH_SIZE),
        METRIC_KEY(IMPORT_EXPORT_SUPPORT),
        GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE),
        GPU_METRIC_KEY(UARCH_VERSION),
        GPU_METRIC_KEY(EXECUTION_UNITS_COUNT),
        GPU_METRIC_KEY(MEMORY_STATISTICS),
    };
    OPENVINO_SUPPRESS_DEPRECATED_END

    return supported_metrics;
}

}  // namespace intel_gpu
}  // namespace ov
