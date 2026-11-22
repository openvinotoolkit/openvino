// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pthread.h>

#include "stack_bounds.hpp"

#if defined(__EMSCRIPTEN__)
#    include <emscripten/stack.h>
#endif

namespace ov::util {

MemoryView query_stack_bounds() {
#if defined(__EMSCRIPTEN__)
    const auto low = static_cast<uintptr_t>(emscripten_stack_get_end());
    const auto high = static_cast<uintptr_t>(emscripten_stack_get_base());
    return {reinterpret_cast<const std::byte*>(low), static_cast<size_t>(high - low)};
#elif defined(__APPLE__)
    // pthread_get_stackaddr_np returns the high end of the stack.
    const auto high = reinterpret_cast<uintptr_t>(pthread_get_stackaddr_np(pthread_self()));
    const size_t size = pthread_get_stacksize_np(pthread_self());
    if (high == 0 || size == 0 || size > high) {
        return {};
    }
    return {reinterpret_cast<const std::byte*>(high - size), size};
#elif defined(__linux__) || defined(__ANDROID__)
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) != 0) {
        return {};
    }
    void* base = nullptr;
    size_t size = 0;
    const int status = pthread_attr_getstack(&attr, &base, &size);
    pthread_attr_destroy(&attr);
    if (status != 0 || base == nullptr || size == 0) {
        return {};
    }
    return {static_cast<const std::byte*>(base), size};
#else
    return {};
#endif
}

uintptr_t current_stack_address() {
#if defined(__EMSCRIPTEN__)
    return static_cast<uintptr_t>(emscripten_stack_get_current());
#else
    // Not the address of a local: AddressSanitizer may place it on a heap "fake stack".
    return reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
#endif
}

}  // namespace ov::util
