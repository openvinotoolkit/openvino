// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file openvino/runtime/threading/executor_manager.hpp
 * @brief A header file for Executor Manager
 */

#pragma once

#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {

/**
 * @interface ExecutorManager
 * @brief Interface for tasks execution manager.
 * This is global point for getting task executor objects by string id.
 * It's necessary in multiple asynchronous requests for having unique executors to avoid oversubscription.
 * E.g. There 2 task executors for CPU device: one - in FPGA, another - in OneDNN. Parallel execution both of them leads
 * to not optimal CPU usage. More efficient to run the corresponding tasks one by one via single executor.
 * @ingroup ov_dev_api_threading
 */
class OPENVINO_API ExecutorManager {
public:
    /**
     * @brief Returns executor by unique identificator
     * @param id An unique identificator of device (Usually string representation of TargetDevice)
     * @return A shared pointer to existing or newly ITaskExecutor
     */
    virtual std::shared_ptr<ITaskExecutor> get_executor(const std::string& id) = 0;

    /// @private
    virtual std::shared_ptr<IStreamsExecutor> get_idle_cpu_streams_executor(
        const IStreamsExecutor::Configuration& config) = 0;

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
     * @brief Set TBB terminate flag
     * @param flag A boolean value:
     * True to terminate tbb during destruction
     * False to not terminate tbb during destruction
     * @return void
     */
    virtual void set_tbb_flag(bool flag) = 0;
    virtual bool get_tbb_flag() = 0;
};

OPENVINO_API std::shared_ptr<ExecutorManager> executor_manager();

}  // namespace ov
