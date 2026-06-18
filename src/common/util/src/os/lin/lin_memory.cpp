// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/mman.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <tuple>
#include <vector>

#include "openvino/util/common_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::util {

namespace {

void madvise_hint(void* ptr, size_t size) noexcept {
    ::madvise(ptr, size, MADV_SEQUENTIAL);
    ::madvise(ptr, size, MADV_WILLNEED);
}

void populate_pages(void* ptr, size_t size, size_t num_threads) noexcept {
    // ptr and size are guaranteed page-aligned by vm_prefetch's precondition.
    const auto page_size = static_cast<size_t>(get_system_page_size());
    const auto chunk_size = std::max<size_t>(align_size_up(size / num_threads, page_size), 1024 * 1024);

    std::vector<std::thread> threads;
    threads.reserve(ceil_div(size, chunk_size));

    for (auto page = reinterpret_cast<const uint8_t*>(ptr), end = page + size; page < end; page += chunk_size) {
        threads.emplace_back([page, chunk_end = std::min(page + chunk_size, end), page_size] {
            volatile uint8_t local = 0;  // prevents the compiler from optimizing the loop away
            for (auto p = page; p < chunk_end; p += page_size) {
                local += *p;
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}

}  // namespace

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
    const auto p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        ec = std::error_code(errno, std::system_category());
        return nullptr;
    }
    ec = {};
    return p;
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

void vm_prefetch(void* ptr, size_t size, size_t num_threads) noexcept {
    assert(ptr != nullptr && size > 0);
    // assert if region is not mmap-backed.

    if (num_threads == 0) {
        // Option 1: OS advisory hints — async, low overhead.
        madvise_hint(ptr, size);
    } else {
        // Option 2: parallel synchronous touch — blocks until every page is resident.
        populate_pages(ptr, size, num_threads);
    }
}

}  // namespace ov::util
