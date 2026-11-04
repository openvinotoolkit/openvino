// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "openvino/core/parallel.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu {

/// Copies @p size bytes in chunks of at least ov::util::default_parallel_io_min_chunk on the shared thread pool.
inline void parallel_memcpy(void* dst, const void* src, size_t size) {
    const size_t max_threads = static_cast<size_t>(parallel_get_max_threads());
    const size_t num_chunks = std::max(size_t{1}, std::min(size / ov::util::default_parallel_io_min_chunk, max_threads));
    const size_t chunk_size = (size + num_chunks - 1) / num_chunks;
    ov::parallel_for(num_chunks, [=](size_t i) {
        const size_t offset = i * chunk_size;
        if (offset < size) {
            std::memcpy(static_cast<char*>(dst) + offset, static_cast<const char*>(src) + offset, std::min(chunk_size, size - offset));
        }
    });
}

}  // namespace ov::intel_gpu
