// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/execution_config.hpp"

namespace ov {
namespace intel_gpu {

ExecutionConfig::ExecutionConfig() {
    set_property(ov::enable_profiling(false));
    set_property(ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::MEDIUM));
    set_property(ov::intel_gpu::hint::queue_priority(ov::hint::Priority::MEDIUM));
    set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));
    set_property(ov::intel_gpu::enable_fusing(true));
    set_property(ov::intel_gpu::optimize_data(true));
    set_property(ov::intel_gpu::enable_memory_pool(true));
    set_property(ov::intel_gpu::allow_static_input_reorder(true));
    set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{}));
    // set_property(ov::intel_gpu::tuning_config({}));
    set_property(ov::intel_gpu::force_implementations(std::vector<std::string>{})); // TODO fix type
    set_property(ov::intel_gpu::partial_build_program(false));
    set_property(ov::intel_gpu::allow_new_shape_infer(false));
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
