// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_executor_manager.hpp
 * @brief A header file for Executor Manager
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <mutex>

#include "threading/ie_itask_executor.hpp"
#include "threading/ie_istreams_executor.hpp"
#include "ie_api.h"

namespace InferenceEngine {

/**
 * @cond
 */
class ExecutorManagerImpl {
public:
    ITaskExecutor::Ptr getExecutor(std::string id);

    IStreamsExecutor::Ptr getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config);

    // for tests purposes
    size_t getExecutorsNumber();

    // for tests purposes
    size_t getIdleCPUStreamsExecutorsNumber();

    void clear(const std::string& id = {});

private:
    std::unordered_map<std::string, ITaskExecutor::Ptr> executors;
    std::vector<std::pair<IStreamsExecutor::Config, IStreamsExecutor::Ptr> > cpuStreamsExecutors;
    std::mutex streamExecutorMutex;
    std::mutex taskExecutorMutex;
};

/**
 * @endcond
 */

/**
 * @brief This is global point for getting task executor objects by string id.
 * It's necessary in multiple asynchronous requests for having unique executors to avoid oversubscription.
 * E.g. There 2 task executors for CPU device: one - in FPGA, another - in MKLDNN. Parallel execution both of them leads
 * to not optimal CPU usage. More efficient to run the corresponding tasks one by one via single executor.
 * @ingroup ie_dev_api_threading
 */
class INFERENCE_ENGINE_API_CLASS(ExecutorManager) {
public:
    /**
     * @brief      Returns a global instance of ExecutorManager
     * @return     The instance.
     */
    static ExecutorManager* getInstance();

    /**
     * @brief A deleted copy constructor
     */
    ExecutorManager(ExecutorManager const&) = delete;

    /**
     * @brief A deleted assignment operator.
     */
    void operator=(ExecutorManager const&) = delete;

    /**
     * @brief Returns executor by unique identificator
     * @param id An unique identificator of device (Usually string representation of TargetDevice)
     * @return A shared pointer to existing or newly ITaskExecutor
     */
    ITaskExecutor::Ptr getExecutor(std::string id);

    /// @private
    IStreamsExecutor::Ptr getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config);

    /**
     * @cond
     */
    size_t getExecutorsNumber();

    size_t getIdleCPUStreamsExecutorsNumber();

    void clear(const std::string& id = {});
    /**
     * @endcond
     */

private:
    ExecutorManager() {}

    ExecutorManagerImpl _impl;

    static std::mutex _mutex;
    static ExecutorManager *_instance;
};

}  // namespace InferenceEngine
