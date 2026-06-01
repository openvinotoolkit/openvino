// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <malloc.h>

#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "openvino/util/memory.hpp"

namespace ov::util {

void* aligned_alloc(size_t size, size_t alignment) noexcept {
    if (alignment == 0) {
        alignment = alignof(std::max_align_t);
    }
    return _aligned_malloc(size, alignment);
}

void aligned_free(void* ptr) noexcept {
    _aligned_free(ptr);
}

bool is_mmap_memory(const void* data) noexcept {
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(data, &mbi, sizeof(mbi)) == 0) {
        return false;
    }
    // Only MEM_MAPPED regions (MapViewOfFile / MapViewOfFile3) qualify as
    // ZE_EXTERNAL_MEMORY_TYPE_FLAG_STANDARD_ALLOCATION on Windows.
    // MEM_PRIVATE (heap, VirtualAlloc) and MEM_IMAGE are not importable.
    return mbi.Type == MEM_MAPPED;
}

}  // namespace ov::util
