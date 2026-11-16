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
/// Falls back to a single std::memcpy when the data fits in one chunk.
inline void parallel_memcpy(void* dst, const void* src, size_t size, size_t min_chunk = ov::util::default_parallel_io_min_chunk) {
    const auto max_threads = static_cast<size_t>(parallel_get_max_threads());
    const size_t chunk_size = std::max({size_t{1}, min_chunk, (size + max_threads - 1) / max_threads});
    if (chunk_size >= size) {
        std::memcpy(dst, src, size);
    } else {
        ov::parallel_for((size + chunk_size - 1) / chunk_size, [=](size_t i) {
            const size_t offset = i * chunk_size;
            std::memcpy(static_cast<char*>(dst) + offset, static_cast<const char*>(src) + offset, std::min(chunk_size, size - offset));
        });
    }
}

}  // namespace ov::intel_gpu
