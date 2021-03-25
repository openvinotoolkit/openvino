// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "ocl_toolkit.h"

namespace cldnn {
namespace gpu {
class command_queues_builder {
public:
    command_queues_builder(const cl::Context& context, const cl::Device& device, const cl_platform_id& platform_id);
    void build();
    void set_throttle_mode(throttle_mode_types throttle, bool extension_support);
    void set_priority_mode(priority_mode_types priority, bool extension_support);
    void set_profiling(bool flag) { _profiling = flag; }
    void set_out_of_order(bool flag) { _out_of_order = flag; }
    queue_type& queue() { return _queue; }
    queue_type queue() const { return _queue; }

private:
    queue_type _queue;
    cl::Context _context;
    cl::Device _device;
    cl_platform_id _platform_id;
    bool _profiling;
    bool _out_of_order;
    priority_mode_types _priority_mode;
    throttle_mode_types _throttle_mode;

    cl_command_queue_properties get_properties();
};
}  // namespace gpu
}  // namespace cldnn
