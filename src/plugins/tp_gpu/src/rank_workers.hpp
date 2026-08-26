// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ov {
namespace tp_gpu {

/// Runs one function per rank, on threads that live as long as the compiled
/// model.
///
/// The obvious implementation is std::async(std::launch::async), and that is
/// what this replaces.  It created one operating-system thread per rank per
/// inference -- at four ranks and forty tokens a second, over a hundred and
/// sixty thread creations every second.  Measured on an 8B model, the first
/// rank's body did not start until 40 us after the launch loop began, and the
/// gap between the first and last rank starting was 47 us at two ranks and
/// 98 us at four.  That spread is the collective's problem, not the thread
/// pool's: every rank has to meet its peers at each of the 64 AllReduce points
/// in a model step, so the group only moves as fast as the rank that started
/// last, and it pays that 64 times per token.
///
/// Rank 0 runs on the calling thread: it needs no handoff, and the caller has
/// nothing else to do while the others work.
class RankWorkers {
public:
    explicit RankWorkers(std::size_t num_ranks) {
        const std::size_t helpers = num_ranks > 0 ? num_ranks - 1 : 0;
        m_workers.reserve(helpers);
        for (std::size_t i = 0; i < helpers; ++i) {
            m_workers.push_back(std::make_unique<Worker>());
        }
        for (std::size_t i = 0; i < helpers; ++i) {
            m_workers[i]->thread = std::thread([this, i] { worker_loop(i); });
        }
    }

    RankWorkers(const RankWorkers&) = delete;
    RankWorkers& operator=(const RankWorkers&) = delete;

    ~RankWorkers() {
        for (auto& w : m_workers) {
            {
                std::lock_guard<std::mutex> lock(w->mtx);
                w->stop = true;
            }
            w->cv.notify_one();
        }
        for (auto& w : m_workers) {
            if (w->thread.joinable()) {
                w->thread.join();
            }
        }
    }

    /// Calls `body(rank)` for every rank and returns once all of them have
    /// finished.  If several ranks throw, the first failure encountered is
    /// rethrown and the rest are dropped -- the group is torn down either way,
    /// and the first failure is the one that explains the others.
    void run(const std::function<void(std::size_t)>& body) {
        m_body = &body;

        for (auto& w : m_workers) {
            {
                std::lock_guard<std::mutex> lock(w->mtx);
                w->error = nullptr;
                w->busy = true;
            }
            w->cv.notify_one();
        }

        std::exception_ptr first_error;
        try {
            body(0);
        } catch (...) {
            first_error = std::current_exception();
        }

        for (auto& w : m_workers) {
            std::unique_lock<std::mutex> lock(w->mtx);
            w->cv.wait(lock, [&] { return !w->busy; });
            if (w->error && !first_error) {
                first_error = w->error;
            }
        }

        m_body = nullptr;
        if (first_error) {
            std::rethrow_exception(first_error);
        }
    }

private:
    struct Worker {
        std::thread thread;
        std::mutex mtx;
        std::condition_variable cv;
        bool busy{false};
        bool stop{false};
        std::exception_ptr error;
    };

    void worker_loop(std::size_t index) {
        Worker& w = *m_workers[index];
        while (true) {
            std::unique_lock<std::mutex> lock(w.mtx);
            w.cv.wait(lock, [&] { return w.busy || w.stop; });
            if (w.stop) {
                return;
            }
            lock.unlock();

            try {
                // m_body was published before busy was set under this same
                // mutex, so acquiring it above is what makes it visible here.
                (*m_body)(index + 1);
            } catch (...) {
                w.error = std::current_exception();
            }

            lock.lock();
            w.busy = false;
            lock.unlock();
            w.cv.notify_one();
        }
    }

    std::vector<std::unique_ptr<Worker>> m_workers;
    const std::function<void(std::size_t)>* m_body{nullptr};
};

}  // namespace tp_gpu
}  // namespace ov
