// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include "details/ie_exception.hpp"

namespace InferenceEngine {

#define MAX_NUMBER_OF_TASKS_IN_QUEUE 10

class TaskSynchronizer {
public:
    typedef std::shared_ptr<TaskSynchronizer> Ptr;

    TaskSynchronizer() : _taskCount(0) {}
    virtual ~TaskSynchronizer() = default;

    virtual void lock() {
        auto taskID = _addTaskToQueue();
        _waitInQueue(taskID);
    }

    virtual void unlock() {
        std::unique_lock<std::mutex> lockTask(_taskMutex);
        if (!_taskQueue.empty()) {
            {
                std::lock_guard<std::mutex> lock(_queueMutex);
                _taskQueue.pop();
            }
            _taskCondVar.notify_all();
        }
    }

    size_t queueSize() const {
        return _taskQueue.size();
    }

private:
    unsigned int _taskCount;
    std::queue<unsigned int> _taskQueue;
    std::mutex _queueMutex;
    std::mutex _taskMutex;
    std::condition_variable _taskCondVar;

protected:
    virtual unsigned int _getTaskID() {
        return _taskCount++;
    }

    virtual unsigned int _addTaskToQueue() {
        std::lock_guard<std::mutex> lock(_queueMutex);
        auto taskID = _getTaskID();
        if (!_taskQueue.empty() && _taskQueue.size() >= MAX_NUMBER_OF_TASKS_IN_QUEUE) {
            THROW_IE_EXCEPTION << "Failed to add more than " << MAX_NUMBER_OF_TASKS_IN_QUEUE << " tasks to queue";
        }
        _taskQueue.push(taskID);
        return taskID;
    }

    virtual void _waitInQueue(unsigned int taskID) {
        std::unique_lock<std::mutex> lock(_taskMutex);
        _taskCondVar.wait(lock, [&]() { return taskID == _taskQueue.front(); });
    }
};

class ScopedSynchronizer {
public:
    explicit ScopedSynchronizer(TaskSynchronizer::Ptr &taskSynchronizer) : _taskSynchronizer(
            taskSynchronizer) {
        _taskSynchronizer->lock();
    }

    ~ScopedSynchronizer() {
        _taskSynchronizer->unlock();
    }

private:
    TaskSynchronizer::Ptr &_taskSynchronizer;
};

}  // namespace InferenceEngine
