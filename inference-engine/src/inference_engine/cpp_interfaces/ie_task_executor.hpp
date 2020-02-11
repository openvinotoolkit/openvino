// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/ie_itask_executor.hpp"
#include "details/ie_exception.hpp"
#include "ie_api.h"

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(TaskExecutor): public ITaskExecutor {
public:
    typedef std::shared_ptr<TaskExecutor> Ptr;

    TaskExecutor(std::string name = "Default");

    ~TaskExecutor();

    void run(Task task) override;

private:
    std::shared_ptr<std::thread> _thread;
    std::mutex _queueMutex;
    std::condition_variable _queueCondVar;
    std::queue<Task> _taskQueue;
    bool _isStopped;
    std::string _name;
};

}  // namespace InferenceEngine
