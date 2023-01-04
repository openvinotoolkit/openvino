// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <thread>

namespace ov {
namespace intel_gpu {

ExecutionConfig::ExecutionConfig() {
    set_default();
}

class InferencePrecisionValidator : public BaseValidator {
public:
    bool is_valid(const ov::Any& v) const override {
        auto precision = v.as<ov::element::Type>();
        return precision == ov::element::f16 || precision == ov::element::f32;
    }
};

void ExecutionConfig::set_default() {
    register_property<PropertyVisibility::PUBLIC>(
        std::make_tuple(ov::device::id, "0"),
        std::make_tuple(ov::enable_profiling, false),
        std::make_tuple(ov::cache_dir, ""),
        std::make_tuple(ov::num_streams, 1),
        std::make_tuple(ov::compilation_num_threads, std::max(1, static_cast<int>(std::thread::hardware_concurrency()))),
        std::make_tuple(ov::hint::inference_precision, ov::element::f16, InferencePrecisionValidator()),
        std::make_tuple(ov::hint::model_priority, ov::hint::Priority::MEDIUM),
        std::make_tuple(ov::hint::performance_mode, ov::hint::PerformanceMode::LATENCY),
        std::make_tuple(ov::hint::num_requests, 0),

        std::make_tuple(ov::intel_gpu::hint::host_task_priority, ov::hint::Priority::MEDIUM),
        std::make_tuple(ov::intel_gpu::hint::queue_throttle, ov::intel_gpu::hint::ThrottleLevel::MEDIUM),
        std::make_tuple(ov::intel_gpu::hint::queue_priority, ov::hint::Priority::MEDIUM),
        std::make_tuple(ov::intel_gpu::enable_loop_unrolling, true),

        // Legacy API properties
        std::make_tuple(ov::intel_gpu::enable_dynamic_batch, false),
        std::make_tuple(ov::intel_gpu::exclusive_async_requests, false),
        std::make_tuple(ov::intel_gpu::nv12_two_inputs, false),
        std::make_tuple(ov::intel_gpu::config_file, ""),
        std::make_tuple(ov::intel_gpu::enable_lp_transformations, false));

    register_property<PropertyVisibility::INTERNAL>(
        std::make_tuple(ov::intel_gpu::max_dynamic_batch, 1),
        std::make_tuple(ov::intel_gpu::queue_type, QueueTypes::out_of_order),
        std::make_tuple(ov::intel_gpu::optimize_data, false),
        std::make_tuple(ov::intel_gpu::enable_memory_pool, true),
        std::make_tuple(ov::intel_gpu::allow_static_input_reorder, false),
        std::make_tuple(ov::intel_gpu::custom_outputs, std::vector<std::string>{}),
        std::make_tuple(ov::intel_gpu::tuning_config, ov::intel_gpu::TuningConfig{}),
        std::make_tuple(ov::intel_gpu::dump_graphs, ""),
        std::make_tuple(ov::intel_gpu::force_implementations, ImplForcingMap{}),
        std::make_tuple(ov::intel_gpu::partial_build_program, false),
        std::make_tuple(ov::intel_gpu::allow_new_shape_infer, false));

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dump_graphs.empty()) {
        set_property(ov::intel_gpu::dump_graphs(debug_config->dump_graphs));
    }
}

void ExecutionConfig::register_property_impl(const std::pair<std::string, ov::Any>& property, PropertyVisibility visibility, BaseValidator::Ptr validator) {
    property_validators[property.first] = validator;
    supported_properties[property.first] = visibility;
    internal_properties[property.first] = property.second;
}

void ExecutionConfig::set_property(const AnyMap& config) {
    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;
        OPENVINO_ASSERT(is_supported(kv.first), "[GPU] Attepmpt to set property ", name, " (", val.as<std::string>(), ") which was not registered!\n");
        OPENVINO_ASSERT(property_validators.at(name)->is_valid(val), "[GPU] Invalid value for property ", name,  ": ", val.as<std::string>());
        internal_properties[name] = val;
    }
}

bool ExecutionConfig::is_supported(const std::string& name) const {
    bool supported = supported_properties.find(name) != supported_properties.end();
    bool has_validator = property_validators.find(name) != property_validators.end();

    return supported && has_validator;
}

bool ExecutionConfig::is_set_by_user(const std::string& name) const {
    return user_properties.find(name) != user_properties.end();
}

void ExecutionConfig::set_user_property(const AnyMap& config) {
    for (auto& kv : config) {
        auto& name = kv.first;
        auto& val = kv.second;
        bool supported = is_supported(name) && supported_properties.at(name) == PropertyVisibility::PUBLIC;
        OPENVINO_ASSERT(supported, "[GPU] Attepmpt to set user property ", name, " (", val.as<std::string>(), ") which was not registered or internal!\n");
        OPENVINO_ASSERT(property_validators.at(name)->is_valid(val), "[GPU] Invalid value for property ", name,  ": ", val.as<std::string>());

        user_properties[kv.first] = kv.second;
    }
}

Any ExecutionConfig::get_property(const std::string& name) const {
    if (user_properties.find(name) != user_properties.end()) {
        return user_properties.at(name);
    }

    OPENVINO_ASSERT(internal_properties.find(name) != internal_properties.end(), "[GPU] Can't get internal property with name ", name);
    return internal_properties.at(name);
}

void ExecutionConfig::apply_performance_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::performance_mode)) {
        const auto mode = get_property(ov::hint::performance_mode);
        if (!is_set_by_user(ov::num_streams)) {
            if (mode == ov::hint::PerformanceMode::LATENCY) {
                set_property(ov::num_streams(1));
            } else if (mode == ov::hint::PerformanceMode::THROUGHPUT) {
                set_property(ov::num_streams(ov::streams::AUTO));
            }
        }
    }

    if (get_property(ov::num_streams) == ov::streams::AUTO) {
        int32_t n_streams = std::max<int32_t>(info.num_ccs, 2);
        set_property(ov::num_streams(n_streams));
    }
}

void ExecutionConfig::apply_priority_hints(const cldnn::device_info& info) {
    if (is_set_by_user(ov::hint::model_priority)) {
        const auto priority = get_property(ov::hint::model_priority);
        if (!is_set_by_user(ov::intel_gpu::hint::queue_priority)) {
            set_property(ov::intel_gpu::hint::queue_priority(priority));
        }
    }
}

void ExecutionConfig::apply_hints(const cldnn::device_info& info) {
    apply_performance_hints(info);
    apply_priority_hints(info);
}

void ExecutionConfig::apply_user_properties(const cldnn::device_info& info) {
    // Copy internal properties before applying hints to ensure that
    // a property set by hint won't be overriden by a value in user config.
    // E.g num_streams=AUTO && hint=THROUGHPUT
    // If we apply hints first and then copy all values from user config to internal one,
    // then we'll get num_streams=AUTO in final config while some integer number is expected.
    for (auto& kv : user_properties) {
        internal_properties[kv.first] = kv.second;
    }
    apply_hints(info);
    if (!is_set_by_user(ov::intel_gpu::enable_lp_transformations)) {
        set_property(ov::intel_gpu::enable_lp_transformations(info.supports_imad || info.supports_immad));
    }
    user_properties.clear();
}

std::string ExecutionConfig::to_string() const {
    std::stringstream s;
    s << "internal properties:\n";
    for (auto& kv : internal_properties) {
        s << "\t" << kv.first << ": " << kv.second.as<std::string>() << std::endl;
    }
    s << "user properties:\n";
    for (auto& kv : user_properties) {
        s << "\t" << kv.first << ": " << kv.second.as<std::string>() << std::endl;
    }
    return s.str();
}

}  // namespace intel_gpu
}  // namespace ov
