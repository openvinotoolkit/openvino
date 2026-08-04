// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_prefetch.hpp"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#include "openvino/util/math_util.hpp"
#include "openvino/util/memory.hpp"

namespace ov::util {

namespace {

// FIFO job queue feeding the shared page-toucher pool: a small pool of long-lived worker threads
// is reused across calls instead of spawning/joining threads per prefetch request. Shared by both
// the Linux and Windows vm_prefetch_async() implementations (see submit_page_toucher_tasks()
// below).
class TaskQueue {
public:
    void push(std::list<std::function<void()>>&& batch) noexcept {
        {
            std::lock_guard lock(m_mutex);
            m_queue.splice(m_queue.end(), batch);
        }
        m_cv.notify_all();
    }

    // Blocks until a job is available, or returns false once the queue is stopped and drained.
    bool wait_and_pop(std::function<void()>& job) noexcept {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] {
            return m_stop || !m_queue.empty();
        });
        if (m_queue.empty()) {
            return false;
        }
        job = std::move(m_queue.front());
        m_queue.pop_front();
        return true;
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
    std::list<std::function<void()>> m_queue;
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

    std::vector<std::future<void>> submit(std::vector<std::function<void()>>&& jobs) {
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
        m_queue.push(std::move(pending));
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
        while (m_queue.wait_and_pop(job)) {
            job();
        }
    }

    TaskQueue m_queue;
    std::vector<std::thread> m_workers;
};

}  // namespace

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
