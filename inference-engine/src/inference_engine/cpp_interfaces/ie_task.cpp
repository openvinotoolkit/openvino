// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <thread>
#include <queue>
#include <ie_profiling.hpp>
#include "details/ie_exception.hpp"
#include "exception2status.hpp"
#include "ie_task_synchronizer.hpp"
#include "ie_task.hpp"
#include "ie_task_executor.hpp"

namespace InferenceEngine {

Task::Task() : _status(TS_INITIAL) {
    _function = [&]() {
        _status = TS_DONE;
        return;
    };
}

Task::Task(std::function<void()> function) : _status(TS_INITIAL), _function(function) {
    if (!function) THROW_IE_EXCEPTION << "Failed to create Task object with null function";
}

Task::Status Task::runNoThrowNoBusyCheck() noexcept {
    IE_PROFILING_AUTO_SCOPE(TaskExecution);
    try {
        _exceptionPtr = nullptr;
        _function();
        setStatus(TS_DONE);
    } catch (...) {
        _exceptionPtr = std::current_exception();
        setStatus(TS_ERROR);
    }
    _isTaskDoneCondVar.notify_all();
    return getStatus();
}

Task::Status Task::runWithSynchronizer(TaskSynchronizer::Ptr &taskSynchronizer) {
    if (occupy()) {
        ScopedSynchronizer scopedSynchronizer(taskSynchronizer);
        runNoThrowNoBusyCheck();
    }
    return getStatus();
}

Task::Status Task::wait(int64_t millis_timeout) {
    _isOnWait = true;
    std::exception_ptr exceptionPtr;
    try {
        std::unique_lock<std::mutex> lock(_taskStatusMutex);
        if (_status != TS_INITIAL) {
            auto predicate = [&]() -> bool { return _status == TS_DONE || _status == TS_ERROR; };
            if (millis_timeout < 0) {
                _isTaskDoneCondVar.wait(lock, predicate);
            } else {
                _isTaskDoneCondVar.wait_for(lock, std::chrono::milliseconds(millis_timeout), predicate);
            }
        }
    } catch (...) {
        exceptionPtr = std::current_exception();
    }
    if (exceptionPtr) std::rethrow_exception(exceptionPtr);
    _isOnWait = false;
    return _status;
}

bool Task::occupy() {
    std::unique_lock<std::mutex> guard(_taskStatusMutex);
    if (_status == Task::TS_BUSY) return false;
    _status = TS_BUSY;
    return true;
}

Task::Status Task::getStatus() {
    std::unique_lock<std::mutex> guard(_taskStatusMutex);
    return _status;
}

void Task::checkException() {
    if (_exceptionPtr) {
        std::rethrow_exception(_exceptionPtr);
    }
}

StatusCode Task::TaskStatus2StatusCode(Task::Status status) {
    switch (status) {
        case Status::TS_DONE:
            return OK;
        case Status::TS_ERROR:
            return GENERAL_ERROR;
        case Status::TS_BUSY:
        case Status::TS_POSTPONED:
            return RESULT_NOT_READY;
        case Status::TS_INITIAL:
            return INFER_NOT_STARTED;
        default:
            THROW_IE_EXCEPTION << "Logic error: unknown state of InferRequest!";
    }
}

void Task::setStatus(Task::Status status) {
    std::unique_lock<std::mutex> guard(_taskStatusMutex);
    _status = status;
}

bool Task::isOnWait() {
    return _isOnWait;
}

}  // namespace InferenceEngine
