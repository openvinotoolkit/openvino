// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/mman.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>

#include "openvino/util/common_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::util {

namespace {

void madvise_hint(void* ptr, size_t size) noexcept {
    madvise(ptr, size, MADV_SEQUENTIAL);
    madvise(ptr, size, MADV_WILLNEED);
}

struct PageToucher {
    const uint8_t* m_begin;
    const uint8_t* m_end;
    const size_t m_page_size;

    void operator()() const noexcept {
        volatile uint8_t local = 0;  // prevents the compiler from optimizing the loop away
        for (auto begin = m_begin; begin < m_end; begin += m_page_size) {
            local += *begin;
        }
    }
};

/**
 * @brief A small, bounded, process-wide thread pool used to run page-population tasks.
 *
 * Rather than spawning dedicated OS threads for every vm_prefetch(_async) call, tasks are
 * queued and picked up by whichever of the pool's fixed worker threads becomes free. This
 * keeps the number of live threads bounded regardless of how many prefetch calls are made
 * concurrently.
 */
class PageTouchThreadPool {
public:
    static PageTouchThreadPool& instance() {
        static PageTouchThreadPool pool;
        return pool;
    }

    PageTouchThreadPool(const PageTouchThreadPool&) = delete;
    PageTouchThreadPool& operator=(const PageTouchThreadPool&) = delete;

    /**
     * @brief Queues @p jobs for execution on the pool and returns a future per job that
     * becomes ready once that job has run.
     */
    std::vector<std::future<void>> submit(std::vector<std::function<void()>>&& jobs) {
        std::vector<std::future<void>> futures;
        futures.reserve(jobs.size());
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (auto& job : jobs) {
                auto task = std::make_shared<std::packaged_task<void()>>(std::move(job));
                futures.push_back(task->get_future());
                m_queue.emplace([task]() {
                    (*task)();
                });
            }
        }
        m_cv.notify_all();
        return futures;
    }

private:
    PageTouchThreadPool()
        : m_workers(std::max<size_t>(1, std::min<size_t>(10, std::thread::hardware_concurrency()))) {
        for (auto& worker : m_workers) {
            worker = std::thread([this]() {
                worker_loop();
            });
        }
    }

    ~PageTouchThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (auto& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this]() {
                    return m_stop || !m_queue.empty();
                });
                if (m_queue.empty()) {
                    if (m_stop) {
                        return;  // fully drained: safe to let this worker exit.
                    }
                    continue;
                }
                job = std::move(m_queue.front());
                m_queue.pop();
            }
            job();
        }
    }

    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stop = false;
};

/**
 * @brief Splits [ptr, ptr + size) into up to @p num_threads page-toucher jobs and submits them
 * to the shared @ref PageTouchThreadPool, returning a future per job.
 */
std::vector<std::future<void>> submit_page_toucher_tasks(void* ptr, size_t size, size_t num_threads) {
    // ptr and size are guaranteed page-aligned by vm_prefetch's precondition.
    const auto page_size = static_cast<size_t>(get_system_page_size());
    const auto chunk_size = std::max<size_t>(align_size_up(size / num_threads, page_size), 1024 * 1024);

    std::vector<std::function<void()>> jobs;
    jobs.reserve(ceil_div(size, chunk_size));

    for (auto first = reinterpret_cast<const uint8_t*>(ptr), last = first + size; first < last; first += chunk_size) {
        jobs.emplace_back(PageToucher{first, std::min(first + chunk_size, last), page_size});
    }
    return PageTouchThreadPool::instance().submit(std::move(jobs));
}

void populate_pages(void* ptr, size_t size, size_t num_threads) {
    // ptr and size are guaranteed page-aligned by vm_prefetch's precondition.
    auto tasks = submit_page_toucher_tasks(ptr, size, num_threads);
    for (auto& task : tasks) {
        task.wait();
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

PrefetchToken vm_prefetch_async(void* ptr, size_t size, size_t num_threads) noexcept {
    assert(ptr != nullptr && size > 0);
    // assert if region is not mmap-backed.

    if (num_threads == 0) {
        // OS advisory hint is cheap and already asynchronous; no background tasks needed.
        madvise_hint(ptr, size);
        return {};
    }
    // Submit the touchers to the shared pool but do not wait on them here — ownership of the
    // futures is transferred to the token, whose destructor (or explicit wait()) waits on them.
    return PrefetchToken(submit_page_toucher_tasks(ptr, size, num_threads));
}

}  // namespace ov::util
