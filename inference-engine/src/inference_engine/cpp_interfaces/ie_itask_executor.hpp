// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "ie_api.h"
#include "ie_task.hpp"

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(ITaskExecutor) {
public:
    typedef std::shared_ptr<ITaskExecutor> Ptr;

    /**
     * @brief Add task for execution and notify working thread about new task to start.
     * @note can be called from multiple threads - tasks will be added to the queue and executed one-by-one in FIFO mode.
     * @param task - shared pointer to the task to start
     *  @return true if succeed to add task, otherwise - false
     */
    virtual bool startTask(Task::Ptr task) = 0;
};

}  // namespace InferenceEngine
