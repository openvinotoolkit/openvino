// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <vector>

namespace ov::util {

/**
 * @brief How deep a prefetch request brings the data in.
 */
enum class PrefetchMode {
    /**
     * Asks the OS to read the range into the page cache (MADV_WILLNEED / PrefetchVirtualMemory).
     * The pages are not mapped into the process, so the resident set size does not grow; the
     * first access costs a minor fault instead of disk I/O.
     */
    readahead,
    /**
     * Maps the range into the process (page population), so the first access costs nothing. Every
     * populated page counts towards the resident set size.
     */
    populate,
};

/**
 * @brief Move-only RAII handle for background prefetch work.
 *
 * Destruction (or an explicit @ref wait) joins the outstanding work, so nothing is ever left
 * running uncontrolled. The token does not keep the prefetched memory alive: the caller must keep
 * that memory valid until the token completes, is destroyed, or its futures are @ref detach "detached".
 */
class PrefetchToken {
public:
    PrefetchToken() noexcept = default;
    explicit PrefetchToken(std::vector<std::future<void>>&& tasks,
                           std::shared_ptr<std::atomic<bool>> cancel_flag = {}) noexcept
        : m_tasks(std::move(tasks)),
          m_cancel(std::move(cancel_flag)) {}

    PrefetchToken(const PrefetchToken&) = delete;
    PrefetchToken& operator=(const PrefetchToken&) = delete;
    PrefetchToken(PrefetchToken&&) noexcept = default;

    PrefetchToken& operator=(PrefetchToken&& other) noexcept {
        if (this != &other) {
            wait();
            m_tasks = std::move(other.m_tasks);
            m_cancel = std::move(other.m_cancel);
        }
        return *this;
    }

    ~PrefetchToken() {
        wait();
    }

    /** @brief Blocks until all the work is finished (or skipped after @ref cancel). */
    void wait() noexcept {
        for (auto& task : m_tasks) {
            if (task.valid()) {
                task.wait();
            }
        }
        m_tasks.clear();
    }

    /** @brief Returns true when no work is pending, never blocks. */
    bool ready() const noexcept {
        for (const auto& task : m_tasks) {
            if (task.valid() && task.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Requests the outstanding work to stop as soon as possible; does not block.
     *
     * Work which has not started yet is skipped, running work stops at the next chunk boundary.
     * Call @ref wait to join it.
     */
    void cancel() noexcept {
        if (m_cancel) {
            m_cancel->store(true, std::memory_order_relaxed);
        }
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
    std::shared_ptr<std::atomic<bool>> m_cancel;
};

/**
 * @brief Starts prefetching the memory range [ptr, ptr + size) in the background.
 *
 * The range does not need to be page aligned: it is extended to the enclosing pages, all of which
 * hold some of the requested bytes and therefore are accessible. Ranges smaller than a page are ignored.
 *
 * @param ptr   Start of the range.
 * @param size  Size of the range in bytes.
 * @param mode  How deep the data is brought in, see @ref PrefetchMode.
 * @return Token to observe, cancel or join the work. Empty if nothing was scheduled.
 */
PrefetchToken prefetch_async(const void* ptr, size_t size, PrefetchMode mode) noexcept;

}  // namespace ov::util
