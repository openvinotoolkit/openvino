// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <thread>
#include "details/ie_exception.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/ie_task.hpp"
#include "cpp_interfaces/ie_task_with_stages.hpp"

namespace InferenceEngine {

StagedTask::StagedTask() : Task(), _stages(0) {}

StagedTask::StagedTask(std::function<void()> function, size_t stages) : Task(function), _stages(stages), _stage(0) {
    if (!function) THROW_IE_EXCEPTION << "Failed to create StagedTask object with null function";
    resetStages();
}

Task::Status StagedTask::runNoThrowNoBusyCheck() noexcept {
    std::lock_guard<std::mutex> lock(_runMutex);
    try {
        _exceptionPtr = nullptr;
        if (_stage) {
            setStatus(TS_POSTPONED);
        }
        _function();
        if (!_stage) {
            setStatus(TS_DONE);
        }
    } catch (...) {
        _exceptionPtr = std::current_exception();
        setStatus(TS_ERROR);
    }

    if (_status != TS_POSTPONED) {
        _isTaskDoneCondVar.notify_all();
    }
    return getStatus();
}

void StagedTask::resetStages() {
    _stage = _stages;
}

void StagedTask::stageDone() {
    if (_stage <= 0) THROW_IE_EXCEPTION << "Failed to make stage done, because it's been already done";
    _stage--;
}

size_t StagedTask::getStage() {
    return _stage;
}

}  // namespace InferenceEngine
