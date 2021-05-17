// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "cldnn/runtime/engine.hpp"

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

private:
    bool _profiling;
    bool _out_of_order;
    priority_mode_types _priority_mode;
    throttle_mode_types _throttle_mode;

    cl_command_queue_properties get_properties();
};

}  // namespace ocl
}  // namespace cldnn
