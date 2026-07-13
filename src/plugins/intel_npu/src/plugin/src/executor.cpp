// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executor.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "intel_npu/common/itt.hpp"

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
            start_worker_locked();
            return;
        }

        while (_activeWorkers < _workersBaseline) {
            start_worker_locked();
        }
    }

    ~AdaptiveThreadExecutor() override {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopped = true;
        }
        _condition.notify_all();
        for (auto& worker : _workers) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
    }

    void run(ov::threading::Task task) override {
        reap_finished_workers();

        if (_workersBaseline == 0 && !_allowWorkerGrowth) {
            try {
                task();
            } catch (...) {
            }
            return;
        }

        {
            std::lock_guard<std::mutex> lock(_mutex);
            _tasks.push(TaskEntry{std::move(task)});

            if (_allowWorkerGrowth) {
                while ((_tasks.size() + _busyWorkers) > _activeWorkers) {
                    start_worker_locked();
                }
            }
        }
        _condition.notify_one();
    }

private:
    struct TaskEntry {
        ov::threading::Task task;
    };

    struct WorkerEntry {
        std::thread thread;
        bool finished = false;
    };

    void reap_finished_workers() {
        std::vector<std::thread> threadsToJoin;
        std::vector<size_t> joinedIndices;

        {
            std::lock_guard<std::mutex> lock(_mutex);
            for (const auto index : _retiredWorkerIndices) {
                if (_workers[index].finished && _workers[index].thread.joinable()) {
                    threadsToJoin.emplace_back(std::move(_workers[index].thread));
                    joinedIndices.emplace_back(index);
                }
            }
            _retiredWorkerIndices.clear();
        }

        for (auto& thread : threadsToJoin) {
            thread.join();
        }

        if (!joinedIndices.empty()) {
            std::lock_guard<std::mutex> lock(_mutex);
            for (const auto index : joinedIndices) {
                _workers[index].finished = false;
                _freeWorkerIndices.emplace_back(index);
            }
        }
    }

    void start_worker_locked() {
        const size_t workerId = _nextWorkerId++;
        size_t workerIndex = 0;
        const bool reuseSlot = !_freeWorkerIndices.empty();
        if (!_freeWorkerIndices.empty()) {
            workerIndex = _freeWorkerIndices.back();
            _freeWorkerIndices.pop_back();
        } else {
            workerIndex = _workers.size();
            _workers.emplace_back();
        }

        try {
            std::thread workerThread([this, workerId, workerIndex] {
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
                                        _workers[workerIndex].finished = true;
                                        _retiredWorkerIndices.emplace_back(workerIndex);
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
                            _workers[workerIndex].finished = true;
                            _retiredWorkerIndices.emplace_back(workerIndex);
                            return;
                        }

                        task = std::move(_tasks.front().task);
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

            _workers[workerIndex].thread = std::move(workerThread);
            _workers[workerIndex].finished = false;
            ++_activeWorkers;
        } catch (...) {
            if (reuseSlot) {
                _freeWorkerIndices.emplace_back(workerIndex);
            } else {
                _workers.pop_back();
            }
            throw;
        }
    }

    const std::string _name;
    const size_t _workersBaseline;
    const bool _allowWorkerGrowth;
    const std::chrono::milliseconds _idleTimeout;
    std::mutex _mutex;
    std::condition_variable _condition;
    std::queue<TaskEntry> _tasks;
    std::vector<WorkerEntry> _workers;
    std::vector<size_t> _retiredWorkerIndices;
    std::vector<size_t> _freeWorkerIndices;
    size_t _nextWorkerId = 0;
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
