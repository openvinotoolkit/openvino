/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "ocl_toolkit.h"

namespace cldnn {
namespace gpu {
class command_queues_builder {
public:
    command_queues_builder(const cl::Context& context, const cl::Device& device, const cl_platform_id& platform_id);
    void build();
    void set_throttle_mode(cldnn_throttle_mode_type throttle, bool extension_support);
    void set_priority_mode(cldnn_priority_mode_type priority, bool extension_support);
    void set_profiling(bool flag) { _profiling = flag; }
    void set_out_of_order(bool flag) { _out_of_order = flag; }
    cl::CommandQueue& queue() { return _queue; }
    cl::CommandQueue queue() const { return _queue; }

private:
    cl::CommandQueue _queue;
    cl::Context _context;
    cl::Device _device;
    cl_platform_id _platform_id;
    bool _profiling;
    bool _out_of_order;
    cldnn_priority_mode_type _priority_mode;
    cldnn_throttle_mode_type _throttle_mode;

    cl_command_queue_properties get_properties();
};
}  // namespace gpu
}  // namespace cldnn
