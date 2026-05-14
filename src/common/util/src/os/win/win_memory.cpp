// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <malloc.h>

#include "openvino/util/memory.hpp"

namespace ov::util {

void* aligned_alloc(size_t size, size_t alignment) noexcept {
    return _aligned_malloc(size, alignment == 0 ? alignof(std::max_align_t) : alignment);
}

void aligned_free(void* ptr) noexcept {
    _aligned_free(ptr);
}

}  // namespace ov::util
