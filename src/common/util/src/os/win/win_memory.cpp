// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <malloc.h>
#include <windows.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <tuple>

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

void* vm_reserve(size_t size, std::error_code& ec) noexcept {
    const auto p = VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_NOACCESS);
    if (p == NULL) {
        ec = std::error_code(GetLastError(), std::system_category());
        return nullptr;
    }
    ec = {};
    return p;
}

void vm_commit(void* ptr, size_t size, std::error_code& ec) noexcept {
    if (VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) == NULL) {
        ec = std::error_code(GetLastError(), std::system_category());
    } else {
        ec = {};
    }
}

void vm_decommit(void* ptr, size_t size) noexcept {
    assert(ptr != nullptr && size > 0);
    std::ignore = VirtualFree(ptr, size, MEM_DECOMMIT);
}

void vm_release(void* ptr, size_t) noexcept {
    assert(ptr != nullptr);
    std::ignore = VirtualFree(ptr, 0, MEM_RELEASE);
}

void vm_prefetch(void* ptr, size_t size, size_t num_threads) noexcept {
    assert(ptr != nullptr && size > 0);
    // CVS-186579
    // assert if region is not mmap-baked.

    if (num_threads == 0) {
        // Option 1: OS advisory hints
    } else {
        // Option 2: parallel synchronous prefault & touch
    }
}

}  // namespace ov::util
