// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/mman.h>

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <tuple>

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

void* vm_reserve(size_t size, std::error_code& ec) noexcept {
    void* result = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (result != MAP_FAILED) {
        ec = {};
    } else {
        ec = std::error_code(errno, std::system_category());
        result = nullptr;
    }
    return result;
}

void vm_commit(void* ptr, size_t size, std::error_code& ec) noexcept {
    if (mprotect(ptr, size, PROT_READ | PROT_WRITE) == -1) {
        ec = std::error_code(errno, std::system_category());
    } else {
        ec = {};
    }
}

void vm_decommit(void* ptr, size_t size) noexcept {
    assert(ptr != nullptr && size > 0);
#if defined(__linux__)
    std::ignore = mprotect(ptr, size, PROT_NONE);
    std::ignore = madvise(ptr, size, MADV_DONTNEED);
#elif defined(__APPLE__) && defined(MADV_FREE_REUSABLE)
    std::ignore = mprotect(ptr, size, PROT_NONE);
    std::ignore = madvise(ptr, size, MADV_FREE_REUSABLE);
#else
    std::ignore = mmap(ptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
#endif
}

void vm_release(void* ptr, size_t size) noexcept {
    assert(ptr != nullptr && size > 0);
    std::ignore = munmap(ptr, size);
}

void vm_prefetch_hint(void* ptr, size_t size) noexcept {
    // OS advisory hints — async, low overhead.
    madvise(ptr, size, MADV_SEQUENTIAL);
    madvise(ptr, size, MADV_WILLNEED);
}

}  // namespace ov::util
