// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_command_queues_builder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <string>

namespace cldnn {
namespace sycl {

command_queues_builder::command_queues_builder()
    : _profiling(false),
      _out_of_order(false),
      _supports_queue_families(false),
      _priority_mode(),
      _throttle_mode() {}

::sycl::property_list command_queues_builder::get_properties(const ::sycl::device& device, uint16_t stream_id) {
    bool profiling = _profiling;
    if (profiling && !device.has(::sycl::aspect::queue_profiling)) {
        profiling = false;
        GPU_DEBUG_INFO << "Requested profiling queue is not supported by current device. Use non-profiling instead\n";
    }

    if (profiling) {
        return _out_of_order ? ::sycl::property_list{::sycl::property::queue::enable_profiling()} :
                               ::sycl::property_list{::sycl::property::queue::enable_profiling(), ::sycl::property::queue::in_order()};
    } else {
        return _out_of_order ? ::sycl::property_list{} :
                               ::sycl::property_list{::sycl::property::queue::in_order()};
    }
}

sycl_queue_type command_queues_builder::build(const ::sycl::context& context, const ::sycl::device& device) {
    static std::atomic<uint16_t> stream_id{0};

    auto properties = get_properties(device, stream_id++);
    try {
        return ::sycl::queue(context, device, properties);
    } catch (const ::sycl::exception& e) {
        OPENVINO_THROW("[GPU] Command queues builder failed to create queue: ", e.what());
    }
}

void command_queues_builder::set_priority_mode(ov::hint::Priority priority, bool extension_support) {
    if (extension_support) {
        _priority_mode = priority;
    }
}

void command_queues_builder::set_throttle_mode(ov::intel_gpu::hint::ThrottleLevel throttle, bool extension_support) {
    if (extension_support) {
        _throttle_mode = throttle;
    }
}

void command_queues_builder::set_supports_queue_families(bool extension_support) {
    _supports_queue_families = extension_support;
}
}  // namespace sycl
}  // namespace cldnn
