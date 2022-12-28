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

void ExecutionConfig::set_default() {
    set_property(ov::device::id("0"));
    set_property(ov::enable_profiling(false));
    set_property(ov::cache_dir(""));
    set_property(ov::num_streams(1));
    set_property(ov::compilation_num_threads(std::max(1, static_cast<int>(std::thread::hardware_concurrency()))));

    set_property(ov::hint::inference_precision(ov::element::f16));
    set_property(ov::hint::model_priority(ov::hint::Priority::MEDIUM));
    set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    set_property(ov::hint::num_requests(0));

    set_property(ov::intel_gpu::hint::host_task_priority(ov::intel_gpu::hint::ThrottleLevel::MEDIUM));
    set_property(ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::MEDIUM));
    set_property(ov::intel_gpu::hint::queue_priority(ov::hint::Priority::MEDIUM));

    set_property(ov::intel_gpu::enable_loop_unrolling(true));

    set_property(ov::intel_gpu::enable_dynamic_batch(false));
    set_property(ov::intel_gpu::max_dynamic_batch(1));
    set_property(ov::intel_gpu::exclusive_async_requests(false));
    set_property(ov::intel_gpu::nv12_two_inputs(false));

    set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));
    set_property(ov::intel_gpu::optimize_data(false));
    set_property(ov::intel_gpu::enable_memory_pool(true));
    set_property(ov::intel_gpu::allow_static_input_reorder(false));
    set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{}));
    set_property(ov::intel_gpu::tuning_config(ov::intel_gpu::TuningConfig{}));
    set_property(ov::intel_gpu::dump_graphs(""));
    set_property(ov::intel_gpu::force_implementations(ImplForcingMap{}));
    set_property(ov::intel_gpu::partial_build_program(false));
    set_property(ov::intel_gpu::allow_new_shape_infer(false));
    set_property(ov::intel_gpu::config_file(""));


    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dump_graphs.empty()) {
        set_property(ov::intel_gpu::dump_graphs(debug_config->dump_graphs));
    }
}

void ExecutionConfig::set_property(const AnyMap& config) {
    for (auto& kv : config) {
        internal_properties[kv.first] = kv.second;
    }
}

bool ExecutionConfig::has_property(std::string name) const {
    return user_properties.find(name) != user_properties.end() || internal_properties.find(name) != internal_properties.end();
}

bool ExecutionConfig::is_set_by_user(std::string name) const {
    return user_properties.find(name) != user_properties.end();
}

void ExecutionConfig::set_user_property(const AnyMap& config) {
    for (auto& kv : config) {
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

ov::AnyMap ExecutionConfig::get_properties() const {
    return internal_properties;
}

void ExecutionConfig::apply_performance_hints() {
    if (is_set_by_user(ov::hint::performance_mode)) {
        const auto mode = get_property(ov::hint::performance_mode);
        bool streams_set = is_set_by_user(ov::num_streams);
        if (!streams_set) {
            if (mode == ov::hint::PerformanceMode::LATENCY) {
                set_property(ov::num_streams(1));
            } else if (mode == ov::hint::PerformanceMode::THROUGHPUT) {
                set_property(ov::num_streams(ov::streams::AUTO));
            }
        }
    }
}
void ExecutionConfig::apply_priority_hints() {
    if (is_set_by_user(ov::hint::model_priority)) {
        const auto priority = get_property(ov::hint::model_priority);
        bool queue_priority_set = is_set_by_user(ov::intel_gpu::hint::queue_priority);
        if (!queue_priority_set) {
            set_property(ov::intel_gpu::hint::queue_priority(priority));
            // TODO: handle cores types
        }
    }
}

void ExecutionConfig::apply_hints() {
    apply_performance_hints();
    apply_priority_hints();
}

void ExecutionConfig::apply_user_properties() {
    apply_hints();
    for (auto& kv : user_properties) {
        internal_properties[kv.first] = kv.second;
    }
    user_properties.clear();
}

std::string ExecutionConfig::to_string() const {
    std::stringstream s;
    s << "Config\n";
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
