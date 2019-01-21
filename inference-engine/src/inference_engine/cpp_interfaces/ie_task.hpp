// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <thread>
#include <queue>
#include "ie_api.h"
#include "details/ie_exception.hpp"
#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/ie_task_synchronizer.hpp"
#include <iostream>
#include "ie_profiling.hpp"

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(Task) {
public:
    typedef std::shared_ptr<Task> Ptr;
    /**
     *@brief Enumeration to hold status of the Task
     */
    typedef enum {
        // Inference failed with some error
                TS_ERROR = -1,
        // Task was never started
                TS_INITIAL = 0,
        // Task is ongoing: waiting in a queue of task or doing inference
                TS_BUSY,
        // Task was finished, result is ready
                TS_DONE,
        // Task was started to executed but was interrupted
        // can change status to BUSY or DONE
                TS_POSTPONED
    } Status;

    Task();

    Task(std::function<void()> function);

    /**
     * @brief Executes the task with catching all exceptions. It doesn't check that task is running
     *  @note Not recommended to call from multiple threads without synchronizaion, because there's no check for task status
     *  @note If task throws exception, it returns TS_ERROR status. Call getExceptionPtr to get pointer to exception.
     * @return Enumeration of the task status: TS_DONE(2) for success
     */
    virtual Status runNoThrowNoBusyCheck() noexcept;

    /**
     * @brief Executes the task in turn, controlled by task synchronizer
     *  @note Can be called from multiple threads - will return TS_BUSY(1) status for the task which is currently running
     * @param taskSynchronizer - shared pointer to the task synchronizer, which ensures thread-safe execution multiple tasks from multiple threads
     * @return Enumeration of the task status: TS_DONE(2) for success
     */
    Status runWithSynchronizer(TaskSynchronizer::Ptr &taskSynchronizer);

    /**
     * @brief Waits for the finishing task. Blocks until specified millis_timeout has elapsed or the task is done, whichever comes first.
     * @param millis_timeout Maximum duration in milliseconds to block for
     *  @note if millis_timeout < 0 it blocks until task is done
     * @return Enumeration of the task status: TS_DONE(2) for success
     */
    Status wait(int64_t millis_timeout);

    /**
     * @brief Occupies task for launching. Makes busy status, if task is not running
     * @return true if occupation succeed, otherwise - false
     */
    bool occupy();

    Status getStatus();

    void checkException();

    static StatusCode TaskStatus2StatusCode(Status status);

    bool isOnWait();

protected:
    void setStatus(Status status);

protected:
    std::function<void()> _function;
    Status _status;
    std::exception_ptr _exceptionPtr = nullptr;
    std::mutex _taskStatusMutex;
    std::condition_variable _isTaskDoneCondVar;

    bool _isOnWait = false;
};

}  // namespace InferenceEngine
