// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/itt.hpp"

#include <cstdlib>

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

void threadName(const char* name) {
    __itt_thread_set_name(name);
}

#else

domain_t domain(char const*) {
    return nullptr;
}

handle_t handle(char const*) {
    return nullptr;
}

void taskBegin(domain_t, handle_t) {}

void taskEnd(domain_t) {}

void threadName(const char*) {}

#endif  // ENABLE_PROFILING_ITT

}  // namespace internal
}  // namespace itt
}  // namespace openvino
