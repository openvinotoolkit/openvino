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
#include "prefetch_pages.hpp"

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
    // assert if region is not mmap-backed.

    if (num_threads == 0) {
        // Option 1: OS advisory hint — async, low overhead. Best-effort; failure is ignored.
        WIN32_MEMORY_RANGE_ENTRY entry{ptr, size};
        ::PrefetchVirtualMemory(::GetCurrentProcess(), 1, &entry, 0);
    } else {
        // Option 2: parallel synchronous touch — blocks until every page is resident.
        populate_pages(ptr, size, num_threads);
    }
}

}  // namespace ov::util
