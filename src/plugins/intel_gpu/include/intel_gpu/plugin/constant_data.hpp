// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include "openvino/runtime/threading/itask_executor.hpp"

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

    auto* dst = static_cast<char*>(destination);
    const auto* src = static_cast<const char*>(source);
    const size_t chunk_size = size / num_chunks;
    std::vector<ov::threading::Task> tasks;
    tasks.reserve(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        const size_t offset = i * chunk_size;
        const size_t count = i + 1 == num_chunks ? size - offset : chunk_size;
        tasks.emplace_back([=] {
            std::memcpy(dst + offset, src + offset, count);
        });
    }
    executor->run_and_wait(tasks);
}

}  // namespace ov::intel_gpu
