// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <map>
#include <atomic>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "ie_plugin.hpp"
#include "cpp/ie_cnn_network.h"
#include "debug_options.h"
#include "inference_engine.hpp"
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/ie_task_executor.hpp>
#include "ie_parallel.hpp"
#include "cldnn_streams_task_executor.h"

namespace CLDNNPlugin {
std::atomic<unsigned int> MultiWorkerTaskExecutor::waitingCounter(0u);

thread_local MultiWorkerTaskContext MultiWorkerTaskExecutor::ptrContext;

MultiWorkerTaskExecutor::MultiWorkerTaskExecutor(const std::vector<InferenceEngine::Task::Ptr>& init_tasks, std::string name) :
        _isStopped(false), _name(name), _initCount(0) {
    for (auto& t : init_tasks) {
        _threads.emplace_back([&, t] {
            // initialization (no contention, every worker thread is doing it's own task)
            t->runNoThrowNoBusyCheck();
            _initCount++;

            while (!_isStopped) {
                bool isQueueEmpty;
                InferenceEngine::Task::Ptr currentTask = nullptr;
                {  // waiting for the new task or for stop signal
                    std::unique_lock<std::mutex> lock(_queueMutex);
                    _queueCondVar.wait(lock, [&]() { return !_taskQueue.empty() || _isStopped; });
                    isQueueEmpty = _taskQueue.empty();
                    if (!isQueueEmpty) {
                        currentTask = _taskQueue.front();
                        _taskQueue.pop();
                        isQueueEmpty = _taskQueue.empty();
                    }
                }
                if (currentTask) {
                    waitingCounter--;
                    currentTask->runNoThrowNoBusyCheck();
                }
                if (_isStopped)
                    break;
                if (isQueueEmpty)  // notify dtor, that all tasks were completed
                    _queueCondVar.notify_all();
            }
        });
    }
    while (_initCount != init_tasks.size()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

MultiWorkerTaskExecutor::~MultiWorkerTaskExecutor() {
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        if (!_taskQueue.empty()) {
            _queueCondVar.wait(lock, [this]() { return _taskQueue.empty(); });
        }
        _isStopped = true;
        _queueCondVar.notify_all();
    }
    for (auto& thread : _threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

bool MultiWorkerTaskExecutor::startTask(InferenceEngine::Task::Ptr task) {
    if (!task->occupy()) return false;
    std::unique_lock<std::mutex> lock(_queueMutex);
    _taskQueue.push(task);
    waitingCounter++;
    _queueCondVar.notify_one();
    return true;
}

};  // namespace CLDNNPlugin
