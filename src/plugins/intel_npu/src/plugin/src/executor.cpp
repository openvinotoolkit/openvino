// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executor.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "intel_npu/common/itt.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"

namespace intel_npu {

namespace {

/**
 * @brief Task executor with adaptive worker growth for NPU plugin jobs.
 *
 * The executor processes tasks submitted through `run()` using an internal queue
 * and a set of worker threads.
 *
 * Behavior depends on constructor parameters:
 * - Fixed-size mode (`allowWorkerGrowth == false`): starts exactly `workers`
 *   threads during construction.
 * - Adaptive mode (`allowWorkerGrowth == true`): starts with one worker and
 *   grows the pool when queued load exceeds active workers.
 *
 * In adaptive mode, workers above the baseline (`workers`) can terminate after
 * `idleTimeout` if no work arrives, reducing idle thread usage.
 *
 * The destructor performs a graceful shutdown by signaling stop, waking workers,
 * and joining all owned threads.
 */
class AdaptiveThreadExecutor final : public ov::threading::ITaskExecutor {
public:
    AdaptiveThreadExecutor(std::string_view name,
                           size_t workers,
                           bool allowWorkerGrowth,
                           std::chrono::milliseconds idleTimeout)
        : _name(name),
          _workersBaseline(workers),
          _allowWorkerGrowth(allowWorkerGrowth),
          _idleTimeout(idleTimeout) {
        std::lock_guard<std::mutex> lock(_mutex);

        if (_allowWorkerGrowth) {
            start_worker_locked(static_cast<int>(_workers.size()));
            return;
        }

        while (_activeWorkers < _workersBaseline) {
            start_worker_locked(static_cast<int>(_workers.size()));
        }
    }

    ~AdaptiveThreadExecutor() override {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopped = true;
        }
        _condition.notify_all();
        for (auto& worker : _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void run(ov::threading::Task task) override {
        if (_workersBaseline == 0 && !_allowWorkerGrowth) {
            task();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(_mutex);
            _tasks.emplace(std::move(task));

            if (_allowWorkerGrowth) {
                while ((_tasks.size() + _busyWorkers) > _activeWorkers) {
                    start_worker_locked(static_cast<int>(_workers.size()));
                }
            }
        }
        _condition.notify_one();
    }

private:
    void start_worker_locked(int workerId) {
        ++_activeWorkers;
        _workers.emplace_back([this, workerId] {
            openvino::itt::threadName(_name + "_" + std::to_string(workerId));
            for (;;) {
                ov::threading::Task task;
                {
                    std::unique_lock<std::mutex> lock(_mutex);
                    while (!_stopped && _tasks.empty()) {
                        if (_activeWorkers > _workersBaseline) {
                            if (!_condition.wait_for(lock, _idleTimeout, [&] {
                                    return _stopped || !_tasks.empty();
                                })) {
                                if (_activeWorkers > _workersBaseline) {
                                    --_activeWorkers;
                                    return;
                                }

                                continue;
                            }
                        } else {
                            _condition.wait(lock, [&] {
                                return _stopped || !_tasks.empty();
                            });
                        }
                    }

                    if (_stopped && _tasks.empty()) {
                        --_activeWorkers;
                        return;
                    }

                    task = std::move(_tasks.front());
                    _tasks.pop();
                    ++_busyWorkers;
                }

                task();

                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    --_busyWorkers;
                }
            }
        });
    }

    const std::string _name;
    const size_t _workersBaseline;
    const bool _allowWorkerGrowth;
    const std::chrono::milliseconds _idleTimeout;
    std::mutex _mutex;
    std::condition_variable _condition;
    std::queue<ov::threading::Task> _tasks;
    std::vector<std::thread> _workers;
    size_t _activeWorkers = 0;
    size_t _busyWorkers = 0;
    bool _stopped = false;
};

}  // namespace

std::shared_ptr<ov::threading::ITaskExecutor> make_executor(std::string_view name,
                                                            size_t workers,
                                                            bool allowWorkerGrowth,
                                                            std::chrono::milliseconds idleTimeout) {
    return std::make_shared<AdaptiveThreadExecutor>(name, workers, allowWorkerGrowth, idleTimeout);
}

}  // namespace intel_npu
