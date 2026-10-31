// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/stream.hpp"

#include <stdexcept>

namespace cldnn {

SyncMethods stream::get_expected_sync_method(const ExecutionConfig& config) {
    auto profiling = config.get_enable_profiling();
    auto queue_type = config.get_queue_type();
    return profiling ? SyncMethods::events : queue_type == QueueTypes::out_of_order ? SyncMethods::barriers
                                                                                    : SyncMethods::none;
}

event::ptr stream::aggregate_events(const std::vector<event::ptr>& events, bool group, bool is_output) {
    if (events.size() == 1 && !is_output) {
        return events[0];
    }

    if (group && !is_output) {
        return group_events(events);
    }

    return events.empty() ? (is_output ? create_user_event(true) : nullptr) : enqueue_marker(events, is_output);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
event::ptr stream::enqueue_onednn(const dnnl::primitive&,
                                  const std::unordered_map<int, dnnl::memory>&,
                                  const std::vector<event::ptr>&,
                                  bool) {
    throw std::runtime_error("[GPU] oneDNN execution on an out-of-order queue is not supported by this stream");
}
#endif

}  // namespace cldnn
