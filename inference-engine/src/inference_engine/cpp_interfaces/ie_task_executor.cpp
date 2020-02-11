// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/ie_task_executor.hpp"

#include <condition_variable>
#include <ie_profiling.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "details/ie_exception.hpp"

namespace InferenceEngine {

TaskExecutor::TaskExecutor(std::string name): _isStopped(false), _name(name) {
    _thread = std::make_shared<std::thread>([&] {
        annotateSetThreadName(("TaskExecutor thread for " + _name).c_str());
        while (!_isStopped) {
            bool isQueueEmpty;
            Task currentTask;
            {  // waiting for the new task or for stop signal
                std::unique_lock<std::mutex> lock(_queueMutex);
                _queueCondVar.wait(lock, [&]() {
                    return !_taskQueue.empty() || _isStopped;
                });
                isQueueEmpty = _taskQueue.empty();
                if (!isQueueEmpty) currentTask = _taskQueue.front();
            }
            if (_isStopped && isQueueEmpty) break;
            if (!isQueueEmpty) {
                currentTask();
                std::unique_lock<std::mutex> lock(_queueMutex);
                _taskQueue.pop();
                isQueueEmpty = _taskQueue.empty();
                if (isQueueEmpty) {
                    // notify dtor, that all tasks were completed
                    _queueCondVar.notify_all();
                }
            }
        }
    });
}

TaskExecutor::~TaskExecutor() {
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        if (!_taskQueue.empty()) {
            _queueCondVar.wait(lock, [this]() {
                return _taskQueue.empty();
            });
        }
        _isStopped = true;
        _queueCondVar.notify_all();
    }
    if (_thread && _thread->joinable()) {
        _thread->join();
        _thread.reset();
    }
}

void TaskExecutor::run(Task task) {
    std::unique_lock<std::mutex> lock(_queueMutex);
    _taskQueue.push(std::move(task));
    _queueCondVar.notify_all();
}

}  // namespace InferenceEngine
