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

#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::util {

namespace {

void madvise_hint(void* ptr, size_t size) noexcept {
    ::madvise(ptr, size, MADV_SEQUENTIAL);
    ::madvise(ptr, size, MADV_WILLNEED);
}

void populate_pages(void* ptr, size_t size, size_t num_threads) noexcept {
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    const auto pages = util::align_size_up(size, page_size) / page_size;

    const auto hw_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    constexpr size_t min_chunk_size = 1024 * 1024;  // 1 MiB per thread minimum
    constexpr size_t max_prefault_threads = 10;
    const auto n_threads =
        std::min({num_threads, hw_threads, pages, max_prefault_threads, std::max<size_t>(1, size / min_chunk_size)});

    const auto base = reinterpret_cast<const uint8_t*>(ptr);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (size_t tid = 0; tid < n_threads; ++tid) {
        threads.emplace_back([&, tid] {
            const size_t begin_page = pages * tid / n_threads;
            const size_t end_page = pages * (tid + 1) / n_threads;
            volatile uint8_t local = 0;  // prevents the compiler from optimizing the loop away
            for (size_t p = begin_page; p < end_page; ++p) {
                const size_t off = p * page_size;
                if (off < size) {
                    local += base[off];
                }
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
    // assert if region is not mmap-baked.

    if (num_threads == 0) {
        // Option 1: OS advisory hints — async, low overhead.
        madvise_hint(ptr, size);
    } else {
        // Option 2: parallel synchronous touch — blocks until every page_size is resident.
        populate_pages(ptr, size, num_threads);
    }
}

}  // namespace ov::util
