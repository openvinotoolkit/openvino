// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/stack_util.hpp"

#include "stack_bounds.hpp"

namespace ov::util {

namespace {

// Cached per thread, as querying the bounds can be expensive.
MemoryView thread_stack() {
    static thread_local const MemoryView stack = query_stack_bounds();
    return stack;
}

}  // namespace

std::optional<size_t> stack_bytes_available() {
    const auto stack = thread_stack();
    const auto low = reinterpret_cast<uintptr_t>(stack.data());
    const auto current = current_stack_address();
    // Not on the thread's own stack (e.g. a coroutine): the remaining size is unknown.
    if (stack.empty() || current < low || current - low >= stack.size()) {
        return std::nullopt;
    }
    return static_cast<size_t>(current - low);
}

bool has_stack_headroom() {
    const auto available = stack_bytes_available();
    return !available || *available > thread_stack().size() / 2;
}

}  // namespace ov::util
