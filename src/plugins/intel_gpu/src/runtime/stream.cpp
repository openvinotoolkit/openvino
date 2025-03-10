// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/stream.hpp"

#include "ocl/ocl_stream.hpp"

#include <stdexcept>

namespace cldnn {

QueueTypes stream::detect_queue_type(engine_types engine_type, void* queue_handle) {
    switch (engine_type) {
        case engine_types::sycl:
        case engine_types::ocl:
            return ocl::ocl_stream::detect_queue_type(queue_handle);
        default: throw std::runtime_error("Invalid engine type");
    }
}

SyncMethods stream::get_expected_sync_method(const ExecutionConfig& config) {
    auto profiling = config.get_enable_profiling();
    auto queue_type = config.get_queue_type();
    return profiling ? SyncMethods::events : queue_type == QueueTypes::out_of_order ? SyncMethods::barriers
                                                                                    : SyncMethods::none;
}

event::ptr stream::aggregate_events(const std::vector<event::ptr>& events, bool group, bool is_output) {
    if (events.size() == 1 && !is_output)
        return events[0];

    if (group && !is_output)
        return group_events(events);

    return events.empty() ? (is_output ? create_user_event(true) : nullptr) : enqueue_marker(events, is_output);
}

}  // namespace cldnn
