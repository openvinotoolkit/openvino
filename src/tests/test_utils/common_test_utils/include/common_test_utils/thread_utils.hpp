// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace ov {
namespace test {
namespace utils {

/// \brief Runs `body` on a new thread with the given stack size.
/// \return The `what()` of the exception thrown by `body`, or an empty string if it returned.
std::string run_with_stack_size(size_t stack_bytes, const std::function<void()>& body);

}  // namespace utils
}  // namespace test
}  // namespace ov
