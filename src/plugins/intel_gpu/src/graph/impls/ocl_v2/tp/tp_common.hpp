// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// First: ze_common.hpp keeps the Level Zero symbol list it needs, which a
// plain include of openvino/zero_api.hpp before it would drop.
#include "ze/ze_stream.hpp"
#include "tp_recording.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov::intel_gpu::ocl {

/// `lhs * rhs` bytes; throws naming `what` when the product overflows.
inline std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* what) {
    OPENVINO_ASSERT(rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs,
                    "[GPU][TP][L0] ",
                    what,
                    " byte count overflow: ",
                    lhs,
                    " * ",
                    rhs);
    return lhs * rhs;
}

/// Duration between two kernel timestamps of one device, in nanoseconds.
inline uint64_t ticks_to_ns(uint64_t start, uint64_t end, uint64_t mask, uint64_t ns_per_tick) {
    const uint64_t s = start & mask;
    const uint64_t e = end & mask;
    const uint64_t delta = (e >= s) ? (e - s) : ((mask + 1 - s) + e);
    return delta * ns_per_tick;
}

/// Aborts the group unless disarmed: a rank that fails leaves the others
/// waiting for something that will never arrive.
struct AbortOnFailure {
    cldnn::ze::tp::GroupState& group;
    uint64_t collective_id;
    const std::string& name;
    uint32_t rank;
    const char* action;  ///< what the rank was doing: "running", "gathering"
    bool armed{true};

    ~AbortOnFailure() {
        if (armed) {
            group.abort_all("[GPU][TP][L0] rank " + std::to_string(rank) + " failed while " + action + " collective " +
                            std::to_string(collective_id) + " (" + name + ")");
        }
    }
};

/// The network's TP channel; throws naming `primitive` when the model is not bound to a TP context.
inline const cldnn::ze::tp::Channel& require_channel(const std::shared_ptr<cldnn::ze::tp::Channel>& channel,
                                                     const char* primitive) {
    OPENVINO_ASSERT(channel != nullptr,
                    "[GPU][TP] ",
                    primitive,
                    " has no TP channel; the model must be compiled or imported with a TP context");
    return *channel;
}

/// The in-order immediate list the model runs on, which a spliced collective is appended to.
inline ze_command_list_handle_t model_queue_of(cldnn::stream& stream, const char* primitive) {
    auto* ze = dynamic_cast<cldnn::ze::ze_stream*>(&stream);
    OPENVINO_ASSERT(ze != nullptr, "[GPU][TP] ", primitive, " expects the Level Zero stream");
    return ze->get_command_list().handle();
}

/// Spliced, the recording lands in the model's in-order queue after the operations that produced the
/// collective's input. Otherwise it runs on the group's own queue, which has no ordering against the model at
/// all, so the model's work is drained first.
inline void drain_unless_spliced(cldnn::stream& stream, const std::vector<cldnn::event::ptr>& events, bool spliced) {
    if (spliced) {
        return;
    }
    if (!events.empty()) {
        stream.wait_for_events(events);
    }
    stream.finish();
}

/// The stream is in-order, so consumers are ordered behind the collective; only a network output needs an event
/// that completes with it.
inline cldnn::event::ptr collective_event(cldnn::stream& stream,
                                          const std::vector<cldnn::event::ptr>& events,
                                          bool is_output) {
    if (is_output) {
        return stream.enqueue_marker({}, true);
    }
    return stream.aggregate_events(events, false, false);
}

}  // namespace ov::intel_gpu::ocl
