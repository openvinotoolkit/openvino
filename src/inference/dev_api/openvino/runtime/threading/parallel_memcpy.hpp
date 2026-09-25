// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/runtime/common.hpp"

namespace ov::threading {
class ITaskExecutor;

/// Copy non-overlapping host-accessible buffers synchronously using the current parallel arena.
/// The caller decides whether the size warrants parallel execution. A zero count is a no-op.
OPENVINO_RUNTIME_API void parallel_memcpy(void* destination, const void* source, size_t count);

/// Copy non-overlapping host-accessible buffers synchronously using an existing task executor.
/// No pool is created. The caller selects the task count; zero or one uses the calling thread.
/// The number of tasks is capped by the byte count. A zero byte count is a no-op.
OPENVINO_RUNTIME_API void parallel_memcpy(void* destination,
                                          const void* source,
                                          size_t count,
                                          ITaskExecutor& executor,
                                          size_t num_tasks);
}  // namespace ov::threading
