// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstring>

#include "openvino/runtime/threading/itask_executor.hpp"
#include "openvino/runtime/threading/parallel_memcpy.hpp"

namespace ov::intel_gpu {

// Keep small copies on the caller and use the existing compilation workers for large constants.
inline void copy_constant_data(void* destination, const void* source, size_t size, ov::threading::ITaskExecutor* executor, size_t max_threads) {
    constexpr size_t min_chunk_size = 4 * 1024 * 1024;
    const size_t num_chunks = std::min(size / min_chunk_size, max_threads);
    if (!executor || num_chunks < 2) {
        if (size != 0)
            std::memcpy(destination, source, size);
        return;
    }

    ov::threading::parallel_memcpy(destination, source, size, *executor, num_chunks);
}

}  // namespace ov::intel_gpu
