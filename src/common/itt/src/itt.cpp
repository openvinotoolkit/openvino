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

static thread_local uint64_t current_region_counter = 0;
static thread_local void* current_region_handle = nullptr;

domain_t domain(const char* name) {
    return reinterpret_cast<domain_t>(__itt_domain_create(name));
}

handle_t handle(const char* name) {
    return reinterpret_cast<handle_t>(__itt_string_handle_create(name));
}

void taskBegin(domain_t d, handle_t t) {
    if (!callStackDepth() || call_stack_depth++ < callStackDepth()) {
        __itt_id parent_id =
            current_region_counter != 0 ? __itt_id_make(current_region_handle, current_region_counter) : __itt_null;
        __itt_task_begin(reinterpret_cast<__itt_domain*>(d),
                         __itt_null,
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
    auto region_counter = nextRegionId();
    current_region_counter = region_counter;
    current_region_handle = reinterpret_cast<void*>(t);
    __itt_id region_id = __itt_id_make(current_region_handle, region_counter);
    __itt_region_begin(reinterpret_cast<__itt_domain*>(d),
                       region_id,
                       __itt_null,
                       reinterpret_cast<__itt_string_handle*>(t));
}

void regionEnd(domain_t d, handle_t t) {
    if (current_region_counter == 0)
        return;

    __itt_id region_id = __itt_id_make(current_region_handle, current_region_counter);
    __itt_region_end(reinterpret_cast<__itt_domain*>(d), region_id);
    current_region_counter = 0;
    current_region_handle = nullptr;
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
