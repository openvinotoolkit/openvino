// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <thread>
#include <vector>

#include "openvino/util/memory.hpp"
#include "openvino/util/memory_prefetch.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

// Touches one byte per page over [m_begin, m_end) to force the pages resident. The volatile
// accumulator keeps the compiler from eliminating the read loop.
struct PageToucher {
    const uint8_t* m_begin;
    const uint8_t* m_end;
    const size_t m_page_size;

    void operator()() const noexcept {
        volatile uint8_t local = 0;
        for (auto begin = m_begin; begin < m_end; begin += m_page_size) {
            local += *begin;
        }
    }
};

/**
 * @brief Pre-fetches a page-aligned, committed VM range into physical memory, blocking until every
 * page is resident.
 *
 * @param ptr          Page-aligned base address of the range.
 * @param size         Multiple of the system page size.
 * @param num_threads  Number of population jobs to split the range into; @c 0 requests only a
 *                     lightweight advisory OS hint instead of touching pages.
 */
void vm_prefetch(void* ptr, size_t size, size_t num_threads) noexcept;

/**
 * @brief Asynchronous variant of @ref vm_prefetch: submits page-population to the shared pool and
 * returns immediately with a @ref PrefetchToken to wait on.
 *
 * @param ptr          Page-aligned base address of the range.
 * @param size         Multiple of the system page size.
 * @param num_threads  Number of population jobs to split the range into; @c 0 requests only a
 *                     lightweight advisory OS hint instead of touching pages and returns an empty
 *                     token.
 *
 * Returns an empty token if the work could not be scheduled.
 */
PrefetchToken vm_prefetch_async(void* ptr, size_t size, size_t num_threads) noexcept;

/**
 * @brief Asks the OS to read [ptr, ptr + size) into the page cache without mapping it into the
 * process (MADV_WILLNEED / PrefetchVirtualMemory), so the resident set size does not grow.
 *
 * @param ptr   Page-aligned base address of the range.
 * @param size  Size of the range in bytes.
 */
void vm_readahead(void* ptr, size_t size) noexcept;

/**
 * @brief Populates the page tables for [ptr, ptr + size) with a single OS call where supported
 * (MADV_POPULATE_READ on Linux 5.14+).
 *
 * @param ptr   Page-aligned base address of the range.
 * @param size  Size of the range in bytes.
 * @return false if the OS has no such facility or the call failed, the caller should touch the pages.
 */
bool vm_populate(void* ptr, size_t size) noexcept;

/**
 * @brief Submits page-population jobs for [ptr, ptr + size) to the shared background thread pool,
 * splitting the range into up to @p num_threads chunks.
 *
 * @param ptr          Page-aligned base address of the range.
 * @param size         Multiple of the system page size.
 * @param num_threads  Number of population jobs to split the range into.
 *
 * Returns an empty vector if the work could not be scheduled (e.g. an allocation failure).
 */
std::vector<std::future<void>> submit_page_toucher_tasks(void* ptr, size_t size, size_t num_threads) noexcept;

/**
 * @brief Clamps [offset, offset + size) to [0, mapping_size) and page-aligns the result, rounding
 * the length up so it is always a page multiple (both ends page-aligned). Returns an empty region
 * (m_length == 0) for a null/empty mapping, an offset at or past the end, or a sub-page request.
 */
inline AlignedRegion clamp_align_region(const void* data, size_t mapping_size, size_t offset, size_t size) noexcept {
    const auto page_size = static_cast<size_t>(get_system_page_size());
    if (data == nullptr || mapping_size == 0 || offset >= mapping_size || size < page_size) {
        return {};
    }
    const auto available = mapping_size - offset;
    const auto raw_len = (size == auto_size) ? available : std::min(size, available);
    auto region = align_region(reinterpret_cast<uintptr_t>(data) + offset, raw_len, page_size);
    region.m_length = align_size_up(region.m_length, page_size);
    return region;
}

/** @brief Upper bound on the shared page-population pool's worker threads. */
inline constexpr size_t max_prefetch_threads = 8;

/**
 * @brief Number of page-population jobs a @p size byte region is split into, honoring the shared
 * parallel-I/O minimum chunk size and the pool worker cap.
 */
inline size_t prefetch_thread_count(size_t size) noexcept {
    const auto pool_cap =
        std::max<size_t>(1, std::min<size_t>(max_prefetch_threads, std::thread::hardware_concurrency()));
    return split_chunk_count(size, default_parallel_io_min_chunk, pool_cap);
}

}  // namespace ov::util
