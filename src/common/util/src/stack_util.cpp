// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/stack_util.hpp"

#include <cstdint>
#include <limits>

#if defined(_WIN32)
#    include <intrin.h>
#    include <windows.h>
#elif defined(__APPLE__) || defined(__linux__) || defined(__ANDROID__)
#    include <pthread.h>
#endif

namespace ov::util {

namespace {

// Address range [low, high) of the current thread's stack; {0, 0} when it cannot be determined.
struct StackBounds {
    uintptr_t low = 0;
    uintptr_t high = 0;
};

// Querying the bounds is expensive (on glibc the main thread's attributes are read from
// /proc/self/maps), so the result is cached per thread.
StackBounds query_stack_bounds() {
#if defined(_WIN32)
#    if (_WIN32_WINNT >= 0x0602)
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    return {static_cast<uintptr_t>(low), static_cast<uintptr_t>(high)};
#    else
    // GetCurrentThreadStackLimits requires Windows 8. The stack is a single reservation, so its
    // allocation base is the lowest address it can grow down to.
    MEMORY_BASIC_INFORMATION mbi{};
    const int probe = 0;
    if (VirtualQuery(&probe, &mbi, sizeof(mbi)) == 0) {
        return {};
    }
    return {reinterpret_cast<uintptr_t>(mbi.AllocationBase), (std::numeric_limits<uintptr_t>::max)()};
#    endif
#elif defined(__APPLE__)
    // pthread_get_stackaddr_np returns the high end of the stack.
    const auto high = reinterpret_cast<uintptr_t>(pthread_get_stackaddr_np(pthread_self()));
    const size_t size = pthread_get_stacksize_np(pthread_self());
    return (high == 0 || size == 0 || size > high) ? StackBounds{} : StackBounds{high - size, high};
#elif (defined(__linux__) || defined(__ANDROID__)) && !defined(__EMSCRIPTEN__)
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
    return {reinterpret_cast<uintptr_t>(base), reinterpret_cast<uintptr_t>(base) + size};
#else
    return {};
#endif
}

// Address within the current stack frame. The address of a local variable is not used: with
// AddressSanitizer's use-after-return detection such a variable can live on a heap "fake stack".
uintptr_t current_stack_address() {
#if defined(_MSC_VER) && !defined(__clang__)
    return reinterpret_cast<uintptr_t>(_AddressOfReturnAddress());
#elif defined(__GNUC__) || defined(__clang__)
    return reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
#else
    const char probe = 0;
    return reinterpret_cast<uintptr_t>(&probe);
#endif
}

}  // namespace

size_t stack_bytes_available() {
    static thread_local const StackBounds bounds = query_stack_bounds();
    const auto current = current_stack_address();
    // Unknown bounds, or a stack that is not the thread's own one (e.g. a coroutine or fiber stack
    // switched to by user code): the remaining size cannot be derived, so report "no limit known".
    if (bounds.low == 0 || current < bounds.low || current >= bounds.high) {
        return (std::numeric_limits<size_t>::max)();
    }
    return static_cast<size_t>(current - bounds.low);
}

}  // namespace ov::util
