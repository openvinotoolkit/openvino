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
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "memory_prefetch.hpp"
#include "openvino/util/math_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

namespace {

void madvise_hint(void* ptr, size_t size) noexcept {
    madvise(ptr, size, MADV_SEQUENTIAL);
    madvise(ptr, size, MADV_WILLNEED);
}

class TaskQueue {
public:
    void push(std::list<std::function<void()>>&& batch) noexcept {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
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
    } catch (...) {
        return {};
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
    if (num_threads == 0) {
        madvise_hint(ptr, size);
    } else {
        PrefetchToken(submit_page_toucher_tasks(ptr, size, num_threads)).wait();
    }
}

PrefetchToken vm_prefetch_async(void* ptr, size_t size) noexcept {
    return PrefetchToken(submit_page_toucher_tasks(ptr, size, prefetch_thread_count(size)));
}

}  // namespace ov::util
