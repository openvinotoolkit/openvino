// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {

ExecutionConfig::ExecutionConfig() {
    set_property(ov::enable_profiling(false));
    set_property(ov::cache_dir(""));
    set_property(ov::num_streams(1));
    set_property(ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::MEDIUM));
    set_property(ov::intel_gpu::hint::queue_priority(ov::hint::Priority::MEDIUM));
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

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->dump_graphs.empty()) {
        set_property(ov::intel_gpu::dump_graphs(debug_config->dump_graphs));
    }
}

void ExecutionConfig::set_property(const AnyMap& config) {
    for (auto& kv : config) {
        properties[kv.first] = kv.second;
    }
}

Any ExecutionConfig::get_property(const std::string& name) const {
    OPENVINO_ASSERT(properties.find(name) != properties.end(), "[GPU] Can't get internal property with name ", name);
    return properties.at(name);
}

}  // namespace intel_gpu
}  // namespace ov
