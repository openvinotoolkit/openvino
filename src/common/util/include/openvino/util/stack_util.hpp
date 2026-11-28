// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <optional>

namespace ov::util {

/// \brief Bytes of stack still available below the caller's frame.
/// \return Remaining bytes, or `std::nullopt` if the stack bounds are unknown.
std::optional<size_t> stack_bytes_available();

/// \brief Checks that more than half of the thread's stack is still free, to bound recursion.
///
/// A fixed reserve is not enough: unwinding and destroying what was built needs stack proportional
/// to the recursion depth.
/// \return `true` if so, or if the stack bounds are unknown.
bool has_stack_headroom();

}  // namespace ov::util
