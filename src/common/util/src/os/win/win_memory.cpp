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

bool is_single_mmap_region(const void* data, size_t size) noexcept {
    if (data == nullptr || size == 0) {
        return false;
    }
    MEMORY_BASIC_INFORMATION mbi_start{}, mbi_end{};
    if (VirtualQuery(data, &mbi_start, sizeof(mbi_start)) == 0 || mbi_start.Type != MEM_MAPPED) {
        return false;
    }
    // Check that the last byte of the range shares the same AllocationBase as the start.
    //
    // The placeholder-based mmap maps the file-backed portion as a SINGLE view
    // (AllocationBase = view_base for every address within it). Two addresses can have
    // different AllocationBase values when:
    //  - The end falls in the anonymous pagefile-backed tail (AllocationBase = tail_start).
    //  - A granule was evicted and re-mapped: each re-mapped piece becomes its own allocation
    //    (AllocationBase = piece_start != original view_base).
    //  - The end is in a wholly separate mmap or heap allocation.
    // All of these cases must prevent ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT.
    const auto* last = static_cast<const char*>(data) + size - 1;
    if (VirtualQuery(last, &mbi_end, sizeof(mbi_end)) == 0 || mbi_end.Type != MEM_MAPPED) {
        return false;
    }
    return mbi_start.AllocationBase == mbi_end.AllocationBase;
}

}  // namespace ov::util
