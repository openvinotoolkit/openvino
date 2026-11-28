// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>

namespace ov::test::utils {

/// \brief Runs `body` on a new thread with the given stack size; rethrows what `body` throws.
void run_with_stack_size(size_t stack_bytes, const std::function<void()>& body);

/// \brief Whether the stack bounds of a thread created by `run_with_stack_size` are known.
bool is_stack_size_observable(size_t stack_bytes);

}  // namespace ov::test::utils
