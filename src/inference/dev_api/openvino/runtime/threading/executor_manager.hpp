// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime Executor Manager
 * @file openvino/runtime/threading/executor_manager.hpp
 */

#pragma once

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {

namespace threading {

/**
 * @interface ExecutorManager
 * @brief Interface for tasks execution manager.
 * This is global point for getting task executor objects by string id.
 * It's necessary in multiple asynchronous requests for having unique executors to avoid oversubscription.
 * E.g. There 2 task executors for CPU device: one - in FPGA, another - in OneDNN. Parallel execution both of them leads
 * to not optimal CPU usage. More efficient to run the corresponding tasks one by one via single executor.
 * @ingroup ov_dev_api_threading
 */
class OPENVINO_RUNTIME_API ExecutorManager {
public:
    /**
     * @brief Returns executor by unique identificator
     * @param id An unique identificator of device (Usually string representation of TargetDevice)
     * @return A shared pointer to existing or newly ITaskExecutor
     */
    virtual std::shared_ptr<ov::threading::ITaskExecutor> get_executor(const std::string& id) = 0;

    /**
     * @brief Returns idle cpu streams executor
     *
     * @param config Streams executor config
     *
     * @return pointer to streams executor config
     */
    virtual std::shared_ptr<ov::threading::IStreamsExecutor> get_idle_cpu_streams_executor(
        const ov::threading::IStreamsExecutor::Config& config) = 0;

    /**
     * @brief Allows to configure executor manager
     *
     * @param properties map with configuration
     */
    virtual void set_property(const ov::AnyMap& properties) = 0;
    /**
     * @brief Returns configuration
     *
     * @param name property name
     *
     * @return Property value
     */
    virtual ov::Any get_property(const std::string& name) const = 0;

    /**
     * @cond
     */
    virtual size_t get_executors_number() const = 0;

    virtual size_t get_idle_cpu_streams_executors_number() const = 0;

    virtual void clear(const std::string& id = {}) = 0;
    /**
     * @endcond
     */
    virtual ~ExecutorManager() = default;

    /**
     * @brief create a temporary executor to execute the specific task
     * @param core_type cpu core type
     * @param task task to be performed
     */
    virtual void execute_task_by_streams_executor(ov::hint::SchedulingCoreType core_type, ov::threading::Task task) = 0;
};

OPENVINO_RUNTIME_API std::shared_ptr<ExecutorManager> executor_manager();
}  // namespace threading
}  // namespace ov
