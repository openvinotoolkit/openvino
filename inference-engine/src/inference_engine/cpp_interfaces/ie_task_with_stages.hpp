// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <thread>
#include <queue>
#include "ie_api.h"
#include "details/ie_exception.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/ie_task.hpp"
#include "ie_task_synchronizer.hpp"

namespace InferenceEngine {

/**
 * This class represents a task which can have several stages
 * and can be migrated from one task executor to another one
 * between stages. This is required to continue execution of the
 * task with special lock for device
 */
class INFERENCE_ENGINE_API_CLASS(StagedTask) : public Task {
public:
    typedef std::shared_ptr<StagedTask> Ptr;

    StagedTask(std::function<void()> function, size_t stages);

    StagedTask();

    Status runNoThrowNoBusyCheck() noexcept override;

    void resetStages();

    void stageDone();

    size_t getStage();

private:
    size_t _stages;
    size_t _stage;
    std::mutex _runMutex;
};


}  // namespace InferenceEngine
