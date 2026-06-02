// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdlib>

#include "openvino/util/memory.hpp"

namespace ov::util {

void* aligned_alloc(size_t size, size_t alignment) noexcept {
    if (alignment == 0) {
        alignment = alignof(std::max_align_t);
    }
    return std::aligned_alloc(alignment, align_size_up(size, alignment));
}

void aligned_free(void* ptr) noexcept {
    std::free(ptr);
}

bool is_mmap_memory(const void* data) noexcept {
    return data != nullptr;
}

bool is_single_mmap_region(const void* data, size_t size) noexcept {
    // Linux mmap always produces a single contiguous file-backed region with no anonymous
    // padding tail; a non-null start pointer with non-zero size is sufficient.
    return data != nullptr && size > 0;
}

}  // namespace ov::util
