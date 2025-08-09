// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/itt.hpp"

#include <atomic>
#include <cstdlib>
#include <vector>

#ifdef ENABLE_PROFILING_ITT
#    include <ittnotify.h>
#endif

namespace openvino {
namespace itt {
namespace internal {

#ifdef ENABLE_PROFILING_ITT

static size_t callStackDepth() {
    static const char* env = std::getenv("OPENVINO_TRACE_DEPTH");
    static const size_t depth = env ? std::strtoul(env, nullptr, 10) : 0;
    return depth;
}

static thread_local uint32_t call_stack_depth = 0;

static uint64_t nextRegionId() {
    static std::atomic<uint64_t> region_id_counter{1};
    return region_id_counter.fetch_add(1, std::memory_order_relaxed);
}

// Thread-local stack to track active region IDs for parent-child relationships
static thread_local std::vector<uint64_t> region_id_stack;

domain_t domain(const char* name) {
    return reinterpret_cast<domain_t>(__itt_domain_create(name));
}

handle_t handle(const char* name) {
    return reinterpret_cast<handle_t>(__itt_string_handle_create(name));
}

void taskBegin(domain_t d, handle_t t) {
    if (!callStackDepth() || call_stack_depth++ < callStackDepth()) {
        __itt_id parent_id = __itt_null;
        if (!region_id_stack.empty()) {
            // Use the current region ID as parent ID for this task
            parent_id = __itt_id_make(nullptr, region_id_stack.back());
        }
        // Create a unique task ID
        __itt_id task_id = __itt_id_make(reinterpret_cast<void*>(t), nextRegionId());
        __itt_task_begin(reinterpret_cast<__itt_domain*>(d),
                         task_id,
                         parent_id,
                         reinterpret_cast<__itt_string_handle*>(t));
    }
}

void taskEnd(domain_t d) {
    if (!callStackDepth() || --call_stack_depth < callStackDepth())
        __itt_task_end(reinterpret_cast<__itt_domain*>(d));
}

void threadName(const char* name) {
    __itt_thread_set_name(name);
}

void regionBegin(domain_t d, handle_t t) {
    auto region_id = nextRegionId();
    region_id_stack.push_back(region_id);
    __itt_id id = __itt_id_make(reinterpret_cast<void*>(t), region_id);
    __itt_region_begin(reinterpret_cast<__itt_domain*>(d), id, __itt_null, reinterpret_cast<__itt_string_handle*>(t));
}

void regionEnd(domain_t d, handle_t t) {
    if (!region_id_stack.empty()) {
        auto region_id = region_id_stack.back();
        region_id_stack.pop_back();
        __itt_id id = __itt_id_make(reinterpret_cast<void*>(t), region_id);
        __itt_region_end(reinterpret_cast<__itt_domain*>(d), id);
    }
}

#else

domain_t domain(const char*) {
    return nullptr;
}

handle_t handle(const char*) {
    return nullptr;
}

void taskBegin(domain_t, handle_t) {}

void taskEnd(domain_t) {}

void threadName(const char*) {}

void regionBegin(domain_t, handle_t) {}

void regionEnd(domain_t, handle_t) {}

#endif  // ENABLE_PROFILING_ITT

}  // namespace internal
}  // namespace itt
}  // namespace openvino
