// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>

namespace ov {
namespace js {

inline std::atomic<uint64_t> lifecycle_trace_sequence{0};

inline bool lifecycle_trace_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("OPENVINO_NODE_LIFECYCLE_TRACE");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

inline void lifecycle_trace(const char* component, const char* event, const void* context, int value = -1) noexcept {
    if (!lifecycle_trace_enabled()) {
        return;
    }

    const auto sequence = lifecycle_trace_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::fprintf(stderr,
                 "[OV_NODE_LIFECYCLE] seq=%llu component=%s event=%s context=%p thread=%zu value=%d\n",
                 static_cast<unsigned long long>(sequence),
                 component,
                 event,
                 const_cast<void*>(context),
                 thread_id,
                 value);
    std::fflush(stderr);
}

}  // namespace js
}  // namespace ov
