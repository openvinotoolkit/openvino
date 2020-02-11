// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ie_api.h"

/**
 * @file ie_itask_executor.hpp
 * @brief A header file for Inference Engine Task Executor Interface
 */

namespace InferenceEngine {
/**
 * @typedef Task
 * @brief Inference Engine Task Executor can use any copyable callable without parameters and output as a task.
 *        It would be wrapped into std::function object
 */
using Task = std::function<void()>;

/**
* @interface ITaskExecutor
* @brief Interface for Task Executor.
*        Inference Engine uses `InferenceEngine::ITaskExecutor` interface to run all asynchronous internal tasks.
*        Different implementations of task executors can be used for different purposes:
*         - To improve cache locality of memory bound CPU tasks some executors can limit task's affinity and maximum
concurrency.
*         - The executor with one worker thread can be used to serialize access to acceleration device.
*         - Immediate task executor can be used to satisfy `InferenceEngine::ITaskExecutor` interface restrictions but
run tasks in current thread.
* @note  Implementation should guaranty thread safety of all methods
* @section Synchronization
*        It is `InferenceEngine::ITaskExecutor` user responsibility to wait for task execution completion.
*        The `c++11` standard way to wait task completion is to use `std::packaged_task` or `std::promise` with
`std::future`.
*        Here is an example of how to use `std::promise` to wait task completion and process task's exceptions:
 * @code
    // std::promise is move only object so to satisfy copy callable constraint we use std::shared_ptr
    auto promise = std::make_shared<std::promise<void>>();
    // When the promise is created we can get std::future to wait the result
    auto future = p->get_future();
    // Rather simple task
    InferenceEngine::Task task = [] {std::cout << "Some Output" << std::endl;};
    // We capture the task and the promise. When the task is executed in the task executor context we munually call
std::promise::set_value() method executorPtr->run([task, promise] { std::exception_ptr currentException; try { task();
        } catch(...) {
            // If there is some exceptions store the pointer to current exception
            currentException = std::current_exception();
        }

        if (nullptr == currentException) {
            promise->set_value();                       //  <-- If there is no problems just call
std::promise::set_value() } else { promise->set_exception(currentException);    //  <-- If there is an exception forward
it to std::future object
        }
    });
    // To wait the task completion we call std::future::wait method
    future.wait();  //  The current thread will be blocked here and wait when std::promise::set_value()
                    //  or std::promise::set_exception() method will be called.
    // If the future store the exception it will be rethrown in std::future::get method
    try {
        future.get()
    } catch(std::exception& e) {
        ProcessError(e);
    }
 * @endcode
 */
class INFERENCE_ENGINE_API_CLASS(ITaskExecutor) {
public:
    using Ptr = std::shared_ptr<ITaskExecutor>;

    virtual ~ITaskExecutor() = default;

    /**
     * @brief Execute InferenceEngine::Task inside task executor context
     * @param task - task to start
     */
    virtual void run(Task task) = 0;

    /**
     * @brief Execute all of the tasks and waits for its completion.
     * Default runAndWait() method implementation uses run() pure virtual method
     * and higher level synchronization primitives from STL.
     * The task is wrapped into std::packaged_task which returns std::future.
     * std::packaged_task will call the task and signal to std::future that the task is finished
     * or the exception is thrown from task
     * Than std::future is used to wait for task execution completion and
     * task exception extraction
     * @NOTE: runAndWait() does not copy or capture tasks!
     * @param tasks - vector of tasks to execute
     */
    virtual void runAndWait(const std::vector<Task>& tasks);
};
}  // namespace InferenceEngine
