// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include "ie_api.h"
#include "cpp_interfaces/ie_itask_executor.hpp"

namespace InferenceEngine {

/**
 * @class ExecutorManagerImpl
 * @brief This class contains implementation of ExecutorManager global instance to provide task executor objects.
 * It helps to create isolated, independent unit tests for the its functionality. Direct usage of ExecutorManager class
 * (which is a singleton) makes it complicated.
 */
class ExecutorManagerImpl {
public:
    ITaskExecutor::Ptr getExecutor(std::string id);

    // for tests purposes
    size_t getExecutorsNumber();

    void clear();

private:
    std::unordered_map<std::string, ITaskExecutor::Ptr> executors;
};

/**
 * @class ExecutorManager
 * @brief This is global point for getting task executor objects by string id.
 * It's necessary in multiple asynchronous requests for having unique executors to avoid oversubscription.
 * E.g. There 2 task executors for CPU device: one - in FPGA, another - in MKLDNN. Parallel execution both of them leads to
 * not optimal CPU usage. More efficient to run the corresponding tasks one by one via single executor.
 */
class INFERENCE_ENGINE_API_CLASS(ExecutorManager) {
public:
    static ExecutorManager *getInstance() {
        if (!_instance) {
            _instance = new ExecutorManager();
        }

        return _instance;
    }

    ExecutorManager(ExecutorManager const &) = delete;

    void operator=(ExecutorManager const &)  = delete;

    /**
     * @brief Returns executor by unique identificator
     * @param id unique identificator of device (Usually string representation of TargetDevice)
     */
    ITaskExecutor::Ptr getExecutor(std::string id);

    // for tests purposes
    size_t getExecutorsNumber();

    // for tests purposes
    void clear();

private:
    ExecutorManager() {}

private:
    ExecutorManagerImpl _impl;
    static ExecutorManager *_instance;
};

}  // namespace InferenceEngine
