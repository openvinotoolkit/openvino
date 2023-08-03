// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/itt.hpp"

#include <cstdlib>
#include <limits>

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

domain_t domain(char const* name) {
    return reinterpret_cast<domain_t>(__itt_domain_create(name));
}

bool is_enabled(domain_t domain) {
    auto d = reinterpret_cast<__itt_domain*>(domain);
    return d == nullptr ? false : d->flags != 0;
}

handle_t handle(char const* name) {
    return reinterpret_cast<handle_t>(__itt_string_handle_create(name));
}

void taskBegin(domain_t d, handle_t t) {
    if (!callStackDepth() || call_stack_depth++ < callStackDepth())
        __itt_task_begin(reinterpret_cast<__itt_domain*>(d),
                         __itt_null,
                         __itt_null,
                         reinterpret_cast<__itt_string_handle*>(t));
}

void taskEnd(domain_t d) {
    if (!callStackDepth() || --call_stack_depth < callStackDepth())
        __itt_task_end(reinterpret_cast<__itt_domain*>(d));
}

void taskBeginEx(domain_t d, handle_t t, unsigned long long timestamp) {
    static const __itt_id s_task_id{std::numeric_limits<uint64_t>::max(), 0, 0};

    if (!callStackDepth() || call_stack_depth++ < callStackDepth())
        __itt_task_begin_ex(reinterpret_cast<__itt_domain*>(d),
                            nullptr,
                            timestamp,
                            s_task_id,
                            __itt_null,
                            reinterpret_cast<__itt_string_handle*>(t));
}

void taskEndEx(domain_t d, unsigned long long timestamp) {
    if (!callStackDepth() || --call_stack_depth < callStackDepth())
        __itt_task_end_ex(reinterpret_cast<__itt_domain*>(d), nullptr, timestamp);
}

void threadName(const char* name) {
    __itt_thread_set_name(name);
}

void setTrack(track_t t) {
    __itt_set_track(reinterpret_cast<__itt_track*>(t));
}

track_t track(char const* name) {
    return reinterpret_cast<track_t>(
        __itt_track_create(nullptr, reinterpret_cast<__itt_string_handle*>(handle(name)), __itt_track_type_normal));
}

timestamp_t timestamp() {
    return static_cast<timestamp_t>(__itt_get_timestamp());
}

#else

domain_t domain(char const*) {
    return nullptr;
}

bool is_enabled(domain_t domain) {
    return false;
}

handle_t handle(char const*) {
    return nullptr;
}

track_t track(char const*) {
    return nullptr;
}

void taskBegin(domain_t, handle_t) {}

void taskEnd(domain_t) {}

void taskBeginEx(domain_t, handle_t, unsigned long long) {}

void taskEndEx(domain_t, unsigned long long) {}

void threadName(const char*) {}

void setTrack(track_t) {}

timestamp_t timestamp() {
    return 0;
}

#endif  // ENABLE_PROFILING_ITT

}  // namespace internal
}  // namespace itt
}  // namespace openvino
