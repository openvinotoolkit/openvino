// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file openvino/runtime/threading/task_executor.hpp
 * @brief A header file for OpenVINO Task Executor Interface
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "openvino/runtime/common.hpp"

namespace ov {
namespace threading {

/**
 * @brief OpenVINO Task Executor can use any copyable callable without parameters and output as a task.
 *        It would be wrapped into std::function object
 * @ingroup ov_dev_api_threading
 */
using Task = std::function<void()>;

/**
* @interface ITaskExecutor
* @ingroup ov_dev_api_threading
* @brief Interface for Task Executor.
*        OpenVINO uses `ov::ITaskExecutor` interface to run all asynchronous internal tasks.
*        Different implementations of task executors can be used for different purposes:
*         - To improve cache locality of memory bound CPU tasks some executors can limit task's affinity and maximum
concurrency.
*         - The executor with one worker thread can be used to serialize access to acceleration device.
*         - Immediate task executor can be used to satisfy `ov::ITaskExecutor` interface restrictions but
run tasks in current thread.
* @note  Implementation should guaranty thread safety of all methods
* @section Synchronization
*        It is `ov::ITaskExecutor` user responsibility to wait for task execution completion.
*        The `c++11` standard way to wait task completion is to use `std::packaged_task` or `std::promise` with
`std::future`.
*        Here is an example of how to use `std::promise` to wait task completion and process task's exceptions:
 * @snippet example_itask_executor.cpp itask_executor:define_pipeline
 */
class OPENVINO_RUNTIME_API ITaskExecutor {
public:
    /**
     * A shared pointer to ITaskExecutor interface
     */
    using Ptr = std::shared_ptr<ITaskExecutor>;

    /**
     * @brief      Destroys the object.
     */
    virtual ~ITaskExecutor() = default;

    /**
     * @brief Execute ov::Task inside task executor context
     * @param task A task to start
     */
    virtual void run(Task task) = 0;

    /**
     * @brief Execute all of the tasks and waits for its completion.
     *        Default run_and_wait() method implementation uses run() pure virtual method
     *        and higher level synchronization primitives from STL.
     *        The task is wrapped into std::packaged_task which returns std::future.
     *        std::packaged_task will call the task and signal to std::future that the task is finished
     *        or the exception is thrown from task
     *        Than std::future is used to wait for task execution completion and
     *        task exception extraction
     * @note run_and_wait() does not copy or capture tasks!
     * @param tasks A vector of tasks to execute
     */
    virtual void run_and_wait(const std::vector<Task>& tasks);
};

}  // namespace threading
}  // namespace ov
