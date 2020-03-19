// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <ie_profiling.hpp>
#include "details/ie_exception.hpp"
#include "ie_task.hpp"
#include "ie_task_executor.hpp"

namespace InferenceEngine {

TaskExecutor::TaskExecutor(std::string name) : _isStopped(false), _name(name) {
    _thread = std::make_shared<std::thread>([&] {
        annotateSetThreadName(("TaskExecutor thread for " + _name).c_str());
        while (!_isStopped) {
            bool isQueueEmpty;
            Task::Ptr currentTask;
            {  // waiting for the new task or for stop signal
                std::unique_lock<std::mutex> lock(_queueMutex);
                _queueCondVar.wait(lock, [&]() { return !_taskQueue.empty() || _isStopped; });
                isQueueEmpty = _taskQueue.empty();
                if (!isQueueEmpty) currentTask = _taskQueue.front();
            }
            if (_isStopped && isQueueEmpty)
                break;
            if (!isQueueEmpty) {
                currentTask->runNoThrowNoBusyCheck();
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
            _queueCondVar.wait(lock, [this]() { return _taskQueue.empty(); });
        }
        _isStopped = true;
        _queueCondVar.notify_all();
    }
    if (_thread && _thread->joinable()) {
        _thread->join();
        _thread.reset();
    }
}

bool TaskExecutor::startTask(Task::Ptr task) {
    if (!task->occupy()) return false;
    std::unique_lock<std::mutex> lock(_queueMutex);
    _taskQueue.push(task);
    _queueCondVar.notify_all();
    return true;
}

}  // namespace InferenceEngine
