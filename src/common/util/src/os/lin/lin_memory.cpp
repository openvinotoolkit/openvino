// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "memory_prefetch.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

namespace {

void madvise_hint(void* ptr, size_t size) noexcept {
    madvise(ptr, size, MADV_SEQUENTIAL);
    madvise(ptr, size, MADV_WILLNEED);
}

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

// Thread-safe queue of population jobs. Owns only the queueing/notification concern.
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

// Process-wide, bounded pool of worker threads that drain a shared TaskQueue. Keeping the worker
// count bounded means repeated prefetch calls reuse the same threads instead of spawning new ones.
class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Enqueueing is all-or-nothing: every throwing operation (task/future creation) happens on
    // local state first, and the finished batch is spliced into the queue with a non-throwing
    // splice. If preparation throws, nothing is enqueued and no job runs.
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
        const auto workers =
            std::max<size_t>(1, std::min<size_t>(max_prefetch_threads, std::thread::hardware_concurrency()));
        m_workers.reserve(workers);
        for (size_t i = 0; i < workers; ++i) {
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

// Splits [ptr, ptr + size) into num_threads page-toucher jobs and submits them to the shared pool.
std::vector<std::future<void>> submit_page_toucher_tasks(void* ptr, size_t size, size_t num_threads) {
    const auto page_size = static_cast<size_t>(get_system_page_size());
    const auto chunk_size =
        std::max<size_t>(align_size_up(size / num_threads, page_size), default_parallel_io_min_chunk);

    std::vector<std::function<void()>> jobs;
    jobs.reserve(ceil_div(size, chunk_size));

    for (auto first = reinterpret_cast<const uint8_t*>(ptr), last = first + size; first < last; first += chunk_size) {
        jobs.emplace_back(PageToucher{first, std::min(first + chunk_size, last), page_size});
    }
    return ThreadPool::instance().submit(std::move(jobs));
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
    std::ignore = munmap(ptr, size);
}

void vm_prefetch(void* ptr, size_t size, size_t num_threads) noexcept {
    if (num_threads == 0) {
        // Advisory-only: let the OS decide, no page touching.
        madvise_hint(ptr, size);
        return;
    }
    // Parallel synchronous touch, blocking until every page is resident. Must not run on a pool
    // worker thread (it would deadlock waiting on the pool). Prefetching is best-effort, so on
    // allocation failure fall back to an advisory hint rather than escaping this noexcept function.
    try {
        PrefetchToken(submit_page_toucher_tasks(ptr, size, num_threads)).wait();
    } catch (...) {
        madvise_hint(ptr, size);
    }
}

PrefetchToken vm_prefetch_async(void* ptr, size_t size) noexcept {
    // Ownership of the futures is transferred to the token; it waits on them on destruction or via
    // an explicit wait(). If submission fails, fall back to an advisory hint and return an empty token.
    try {
        return PrefetchToken(submit_page_toucher_tasks(ptr, size, prefetch_thread_count(size)));
    } catch (...) {
        madvise_hint(ptr, size);
        return PrefetchToken{};
    }
}

}  // namespace ov::util
