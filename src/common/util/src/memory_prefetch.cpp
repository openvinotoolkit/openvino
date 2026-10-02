// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_prefetch.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#include "openvino/util/math_util.hpp"
#include "openvino/util/memory.hpp"

namespace ov::util {

namespace {

// Job priority in the shared pool.
enum class Priority {
    high,  // the data is needed soon (page population), may be waited for
    low,   // background I/O hints (readahead), must never delay high priority jobs
};

// Two-level FIFO job queue feeding the shared page-toucher pool: a small pool of long-lived worker
// threads is reused across calls instead of spawning/joining threads per prefetch request. Shared by
// both the Linux and Windows implementations. High priority jobs are always taken first; at most
// max_low_running workers run low priority jobs at a time, so the others stay free for the high ones.
class TaskQueue {
public:
    static constexpr size_t max_low_running = 1;

    void push(std::list<std::function<void()>>&& batch, Priority priority) noexcept {
        {
            std::lock_guard lock(m_mutex);
            auto& queue = (priority == Priority::high) ? m_high : m_low;
            queue.splice(queue.end(), batch);
        }
        m_cv.notify_all();
    }

    // Blocks until a job is available, or returns false once the queue is stopped and drained.
    bool wait_and_pop(std::function<void()>& job, Priority& priority) noexcept {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] {
            return m_stop || !m_high.empty() || (!m_low.empty() && m_low_running < max_low_running);
        });
        if (!m_high.empty()) {
            job = std::move(m_high.front());
            m_high.pop_front();
            priority = Priority::high;
            return true;
        }
        if (!m_low.empty() && m_low_running < max_low_running) {
            job = std::move(m_low.front());
            m_low.pop_front();
            ++m_low_running;
            priority = Priority::low;
            return true;
        }
        // Stopped: drain the low priority jobs too, their futures must be satisfied.
        if (!m_low.empty()) {
            job = std::move(m_low.front());
            m_low.pop_front();
            ++m_low_running;
            priority = Priority::low;
            return true;
        }
        return false;
    }

    void done(Priority priority) noexcept {
        if (priority == Priority::low) {
            {
                std::lock_guard lock(m_mutex);
                --m_low_running;
            }
            m_cv.notify_all();
        }
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::list<std::function<void()>> m_high;
    std::list<std::function<void()>> m_low;
    size_t m_low_running = 0;
    bool m_stop = false;
};

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    std::vector<std::future<void>> submit(std::vector<std::function<void()>>&& jobs,
                                          Priority priority = Priority::high) {
        std::vector<std::future<void>> futures;
        futures.reserve(jobs.size());
        std::list<std::function<void()>> pending;
        for (auto& job : jobs) {
            auto task = std::make_shared<std::packaged_task<void()>>(std::move(job));
            futures.push_back(task->get_future());
            pending.emplace_back([task]() {
                (*task)();
            });
        }
        m_queue.push(std::move(pending), priority);
        return futures;
    }

private:
    ThreadPool() {
        const auto workers_count =
            std::max<size_t>(1, std::min<size_t>(max_prefetch_threads, std::thread::hardware_concurrency()));
        m_workers.reserve(workers_count);
        for (size_t i = 0; i < workers_count; ++i) {
            m_workers.emplace_back([this]() {
                worker_loop();
            });
        }
    }

    ~ThreadPool() {
        m_queue.stop();
        for (auto& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void worker_loop() noexcept {
        std::function<void()> job;
        Priority priority = Priority::high;
        while (m_queue.wait_and_pop(job, priority)) {
            job();
            job = nullptr;
            m_queue.done(priority);
        }
    }

    TaskQueue m_queue;
    std::vector<std::thread> m_workers;
};

// Granularity at which a running prefetch job checks for cancellation.
constexpr size_t cancel_check_chunk = default_parallel_io_min_chunk;
// Readahead only submits I/O; jobs are kept moderate so a single low priority worker stays responsive.
constexpr size_t readahead_job_size = 16 * one_mib;

void populate_range(uint8_t* first, uint8_t* last, size_t page_size, const std::atomic<bool>& cancel) noexcept {
    for (; first < last; first += cancel_check_chunk) {
        if (cancel.load(std::memory_order_relaxed)) {
            return;
        }
        const auto chunk_end =
            (static_cast<size_t>(last - first) > cancel_check_chunk) ? first + cancel_check_chunk : last;
        if (!vm_populate(first, static_cast<size_t>(chunk_end - first))) {
            PageToucher{first, chunk_end, page_size}();
        }
    }
}

void readahead_range(uint8_t* first, uint8_t* last, const std::atomic<bool>& cancel) noexcept {
    if (!cancel.load(std::memory_order_relaxed)) {
        vm_readahead(first, static_cast<size_t>(last - first));
    }
}

}  // namespace

PrefetchToken prefetch_async(const void* ptr, size_t size, PrefetchMode mode) noexcept {
    const auto page_size = static_cast<size_t>(get_system_page_size());
    if (ptr == nullptr || size < page_size) {
        return {};
    }
    const auto raw_begin = reinterpret_cast<uintptr_t>(ptr);
    if (raw_begin > std::numeric_limits<uintptr_t>::max() - size - page_size) {
        return {};
    }
    const auto begin = align_size_down(raw_begin, page_size);
    const auto end = align_size_up(raw_begin + size, page_size);
    const auto length = static_cast<size_t>(end - begin);

    try {
        auto cancel = std::make_shared<std::atomic<bool>>(false);
        const auto job_size = (mode == PrefetchMode::readahead)
                                  ? readahead_job_size
                                  : std::max<size_t>(align_size_up(length / prefetch_thread_count(length), page_size),
                                                     default_parallel_io_min_chunk);

        std::vector<std::function<void()>> jobs;
        jobs.reserve(ceil_div(length, job_size));
        for (auto first = reinterpret_cast<uint8_t*>(begin), last = first + length; first < last; first += job_size) {
            const auto job_end = (static_cast<size_t>(last - first) > job_size) ? first + job_size : last;
            if (mode == PrefetchMode::readahead) {
                jobs.emplace_back([first, job_end, cancel]() {
                    readahead_range(first, job_end, *cancel);
                });
            } else {
                jobs.emplace_back([first, job_end, page_size, cancel]() {
                    populate_range(first, job_end, page_size, *cancel);
                });
            }
        }
        const auto priority = (mode == PrefetchMode::readahead) ? Priority::low : Priority::high;
        return PrefetchToken(ThreadPool::instance().submit(std::move(jobs), priority), std::move(cancel));
    } catch (...) {
        // Prefetching is only a hint: allocation failures leave the data to be faulted in on access.
        return {};
    }
}

std::vector<std::future<void>> submit_page_toucher_tasks(void* ptr, size_t size, size_t num_threads) noexcept {
    try {
        const auto page_size = static_cast<size_t>(get_system_page_size());
        const auto chunk_size =
            std::max<size_t>(align_size_up(size / num_threads, page_size), default_parallel_io_min_chunk);

        std::vector<std::function<void()>> jobs;
        jobs.reserve(ceil_div(size, chunk_size));

        for (auto first = reinterpret_cast<const uint8_t*>(ptr), last = first + size; first < last;
             first += chunk_size) {
            jobs.emplace_back(PageToucher{first, std::min(first + chunk_size, last), page_size});
        }
        return ThreadPool::instance().submit(std::move(jobs));
    } catch (const std::bad_alloc&) {
        // Job/future/packaged_task allocation failed under memory pressure.
        return {};
    } catch (const std::length_error&) {
        // vector::reserve()'s requested capacity exceeded max_size() (e.g. a pathological
        // ptr/size/num_threads combination producing an absurd chunk count).
        return {};
    }
}

}  // namespace ov::util
