// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_itask_executor.hpp
 * @brief A header file for Inference Engine Task Executor Interface
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ie_api.h"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace InferenceEngine {

/**
 * @brief Inference Engine Task Executor can use any copyable callable without parameters and output as a task.
 *        It would be wrapped into std::function object
 * @ingroup ie_dev_api_threading
 */
using Task = ov::threading::Task;

/**
* @interface ITaskExecutor
* @ingroup ie_dev_api_threading
* @brief Interface for Task Executor.
*        Inference Engine uses `InferenceEngine::ITaskExecutor` interface to run all asynchronous internal tasks.
*        Different implementations of task executors can be used for different purposes:
*         - To improve cache locality of memory bound CPU tasks some executors can limit task's affinity and maximum
concurrency.
*         - The executor with one worker thread can be used to serialize access to acceleration device.
*         - Immediate task executor can be used to satisfy `InferenceEngine::ITaskExecutor` interface restrictions but
run tasks in current thread.
* @note  Implementation should guaranty thread safety of all methods
*        It is `InferenceEngine::ITaskExecutor` user responsibility to wait for task execution completion.
*        The `c++11` standard way to wait task completion is to use `std::packaged_task` or `std::promise` with
`std::future`.
*        Here is an example of how to use `std::promise` to wait task completion and process task's exceptions:
 * @snippet example_itask_executor.cpp itask_executor:define_pipeline
 */
class INFERENCE_ENGINE_API_CLASS(ITaskExecutor) : virtual public ov::threading::ITaskExecutor {
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
     * @brief Execute all of the tasks and waits for its completion.
     *        Default runAndWait() method implementation uses run() pure virtual method
     *        and higher level synchronization primitives from STL.
     *        The task is wrapped into std::packaged_task which returns std::future.
     *        std::packaged_task will call the task and signal to std::future that the task is finished
     *        or the exception is thrown from task
     *        Than std::future is used to wait for task execution completion and
     *        task exception extraction
     * @note runAndWait() does not copy or capture tasks!
     * @param tasks A vector of tasks to execute
     */
    virtual void runAndWait(const std::vector<Task>& tasks);
};

}  // namespace InferenceEngine
