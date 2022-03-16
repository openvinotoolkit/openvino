// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_executor_manager.hpp
 * @brief A header file for Executor Manager
 */

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "threading/ie_istreams_executor.hpp"
#include "threading/ie_itask_executor.hpp"

namespace InferenceEngine {

/**
 * @interface ExecutorManager
 * @brief Interface for tasks execution manager.
 * This is global point for getting task executor objects by string id.
 * It's necessary in multiple asynchronous requests for having unique executors to avoid oversubscription.
 * E.g. There 2 task executors for CPU device: one - in FPGA, another - in OneDNN. Parallel execution both of them leads
 * to not optimal CPU usage. More efficient to run the corresponding tasks one by one via single executor.
 * @ingroup ie_dev_api_threading
 */
class INFERENCE_ENGINE_API_CLASS(ExecutorManager) {
public:
    /**
     * A shared pointer to ExecutorManager interface
     */
    using Ptr = std::shared_ptr<ExecutorManager>;

    /**
     * @brief Returns executor by unique identificator
     * @param id An unique identificator of device (Usually string representation of TargetDevice)
     * @return A shared pointer to existing or newly ITaskExecutor
     */
    virtual ITaskExecutor::Ptr getExecutor(const std::string& id) = 0;

    /// @private
    virtual IStreamsExecutor::Ptr getIdleCPUStreamsExecutor(const IStreamsExecutor::Config& config) = 0;

    /**
     * @cond
     */
    virtual size_t getExecutorsNumber() const = 0;

    virtual size_t getIdleCPUStreamsExecutorsNumber() const = 0;

    virtual void clear(const std::string& id = {}) = 0;
    /**
     * @endcond
     */

    virtual ~ExecutorManager() = default;

    /**
     * @brief      Returns a global instance of ExecutorManager
     * @return     The instance.
     */
    INFERENCE_ENGINE_DEPRECATED("Use IInferencePlugin::executorManager() instead")
    static ExecutorManager* getInstance();
};

INFERENCE_ENGINE_API_CPP(ExecutorManager::Ptr) executorManager();

}  // namespace InferenceEngine
