// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include "ie_api.h"
#include "details/ie_exception.hpp"
#include "cpp_interfaces/ie_task_synchronizer.hpp"
#include "cpp_interfaces/ie_task.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/ie_itask_executor.hpp"

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(TaskExecutor) : public ITaskExecutor {
public:
    typedef std::shared_ptr<TaskExecutor> Ptr;

    TaskExecutor(std::string name = "Default");

    ~TaskExecutor();

    /**
     * @brief Add task for execution and notify working thread about new task to start.
     * @note can be called from multiple threads - tasks will be added to the queue and executed one-by-one in FIFO mode.
     * @param task - shared pointer to the task to start
     *  @return true if succeed to add task, otherwise - false
     */
    bool startTask(Task::Ptr task) override;

private:
    std::shared_ptr<std::thread> _thread;
    std::mutex _queueMutex;
    std::condition_variable _queueCondVar;
    std::queue<Task::Ptr> _taskQueue;
    bool _isStopped;
    std::string _name;
};

}  // namespace InferenceEngine
