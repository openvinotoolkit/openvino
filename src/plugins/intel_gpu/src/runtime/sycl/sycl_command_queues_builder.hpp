// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl_common.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace cldnn {
namespace sycl {

class command_queues_builder {
public:
    command_queues_builder();
    sycl_queue_type build(const ::sycl::context& context, const ::sycl::device& device);
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
    ::sycl::property_list get_properties(const ::sycl::device& device, uint16_t stream_id = 0);
};

}  // namespace sycl
}  // namespace cldnn
