// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/util/memory.hpp"

namespace ov::util {

/// \brief OS-specific. The current thread's stack (lowest address and size), empty if unknown.
MemoryView query_stack_bounds();

/// \brief OS-specific. An address in the caller's stack frame.
uintptr_t current_stack_address();

}  // namespace ov::util
