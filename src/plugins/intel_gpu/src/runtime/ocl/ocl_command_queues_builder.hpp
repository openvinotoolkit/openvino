// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace cldnn {
namespace ocl {

class command_queues_builder {
public:
    command_queues_builder();
    ocl_queue_type build(const cl::Context& context, const cl::Device& device);
    void set_throttle_mode(throttle_mode_types throttle, bool extension_support);
    void set_priority_mode(priority_mode_types priority, bool extension_support);
    void set_profiling(bool flag) { _profiling = flag; }
    void set_out_of_order(bool flag) { _out_of_order = flag; }
    void set_supports_queue_families(bool extension_support);

private:
    bool _profiling;
    bool _out_of_order;
    bool _supports_queue_families;
    priority_mode_types _priority_mode;
    throttle_mode_types _throttle_mode;

    std::vector<cl_queue_properties> get_properties(const cl::Device& device, uint16_t stream_id = 0);
};

}  // namespace ocl
}  // namespace cldnn
