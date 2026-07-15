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
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

class PrefetchToken;

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
 * returns immediately with a @ref PrefetchToken to wait on. Returns an empty token if the work
 * could not be scheduled.
 */
PrefetchToken vm_prefetch_async(void* ptr, size_t size) noexcept;


/**
 * @brief Move-only RAII handle for background page-population started by @ref vm_prefetch_async.
 *
 * Destruction (or an explicit @ref wait) joins the outstanding work, so nothing is ever left
 * running uncontrolled. The token does not keep the populated memory alive: the caller must keep
 * that memory valid until the token completes, is destroyed, or its futures are @ref detach "detached".
 */
class PrefetchToken {
public:
    PrefetchToken() noexcept = default;
    explicit PrefetchToken(std::vector<std::future<void>>&& tasks) noexcept : m_tasks(std::move(tasks)) {}

    PrefetchToken(const PrefetchToken&) = delete;
    PrefetchToken& operator=(const PrefetchToken&) = delete;
    PrefetchToken(PrefetchToken&&) noexcept = default;

    PrefetchToken& operator=(PrefetchToken&& other) noexcept {
        if (this != &other) {
            wait();
            m_tasks = std::move(other.m_tasks);
        }
        return *this;
    }

    ~PrefetchToken() {
        wait();
    }

    void wait() noexcept {
        for (auto& task : m_tasks) {
            if (task.valid()) {
                task.wait();
            }
        }
        m_tasks.clear();
    }

    std::vector<std::future<void>> detach() noexcept {
        auto tasks = std::move(m_tasks);
        m_tasks.clear();
        return tasks;
    }

    bool valid() const noexcept {
        return !m_tasks.empty();
    }

    explicit operator bool() const noexcept {
        return valid();
    }

private:
    std::vector<std::future<void>> m_tasks;
};

/**
 * @brief Clamps [offset, offset + size) to [0, mapping_size) and page-aligns the result. Returns an
 * empty region (m_length == 0) for a null/empty mapping, an offset at or past the end, or a
 * sub-page request.
 */
inline AlignedRegion clamp_align_region(const void* data, size_t mapping_size, size_t offset, size_t size) noexcept {
    const auto page_size = static_cast<size_t>(get_system_page_size());
    if (data == nullptr || mapping_size == 0 || offset >= mapping_size || size < page_size) {
        return {};
    }
    const auto available = mapping_size - offset;
    const auto raw_len = (size == auto_size) ? available : std::min(size, available);
    return align_region(reinterpret_cast<uintptr_t>(data) + offset, raw_len, page_size);
}

/** @brief Aligned region and page-aligned size shared between the sync and async hint_prefetch(). */
struct PrefetchPlan {
    uintptr_t m_address = 0;
    size_t m_aligned_size = 0;
};

/**
 * @brief Computes the region and page-aligned size for a hint_prefetch()/hint_prefetch_async()
 * call, or an empty plan (m_aligned_size == 0) when the region is below the parallel-I/O threshold
 * (a real population pass would not be worth it).
 */
inline PrefetchPlan make_prefetch_plan(const void* data, size_t mapping_size, size_t offset, size_t size) noexcept {
    if (const auto region = clamp_align_region(data, mapping_size, offset, size);
        region.m_length > default_parallel_io_threshold) {
        return {region.m_address, align_size_up(region.m_length, static_cast<size_t>(get_system_page_size()))};
    }
    return {};
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
