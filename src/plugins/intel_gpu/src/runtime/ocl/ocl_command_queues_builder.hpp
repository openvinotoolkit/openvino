// Copyright (C) 2018-2025 Intel Corporation
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
    void set_throttle_mode(ov::intel_gpu::hint::ThrottleLevel throttle, bool extension_support);
    void set_priority_mode(ov::hint::Priority priority, bool extension_support);
    void set_profiling(bool flag) { _profiling = flag; }
    void set_out_of_order(bool flag) { _out_of_order = flag; }
    void set_supports_queue_families(bool extension_support);

private:
    bool _profiling;
    bool _out_of_order;
    bool _supports_queue_families;
    std::optional<ov::hint::Priority> _priority_mode;
    std::optional<ov::intel_gpu::hint::ThrottleLevel> _throttle_mode;
#if CL_TARGET_OPENCL_VERSION >= 200
    std::vector<cl_queue_properties> get_properties(const cl::Device& device, uint16_t stream_id = 0);
#else
    cl_command_queue_properties get_properties(const cl::Device& device, uint16_t stream_id = 0);
#endif
};

}  // namespace ocl
}  // namespace cldnn
