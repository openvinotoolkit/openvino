// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <intrin.h>
#include <windows.h>

#include "stack_bounds.hpp"

namespace ov::util {

MemoryView query_stack_bounds() {
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    return {reinterpret_cast<const std::byte*>(low), static_cast<size_t>(high - low)};
}

uintptr_t current_stack_address() {
#if defined(_MSC_VER)
    return reinterpret_cast<uintptr_t>(_AddressOfReturnAddress());
#else
    return reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
#endif
}

}  // namespace ov::util
