// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <cpp_interfaces/ie_task_executor.hpp>
#include <ie_common.h>
#include "task_tests_utils.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class TaskExecutorTests : public ::testing::Test {};

TEST_F(TaskExecutorTests, canCreateTaskExecutor) {
    EXPECT_NO_THROW(std::make_shared<TaskExecutor>());
}

TEST_F(TaskExecutorTests, canCatchException) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    auto task = std::make_shared<Task>([]() {
        THROW_IE_EXCEPTION;
    });
    taskExecutor->startTask(task);
    auto status = task->wait(-1);
    ASSERT_EQ(status, Task::Status::TS_ERROR);
    EXPECT_THROW(task->checkException(), InferenceEngineException);
}

TEST_F(TaskExecutorTests, canRunDefaultTask) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    auto defaultTask = std::make_shared<Task>();
    taskExecutor->startTask(defaultTask);
    auto status = defaultTask->wait(-1);
    ASSERT_EQ(status, Task::Status::TS_DONE);
}

TEST_F(TaskExecutorTests, canRunDefaultTaskWithoutStop) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    auto defaultTask = std::make_shared<Task>();
    taskExecutor->startTask(defaultTask);
    auto status = defaultTask->wait(-1);
    ASSERT_EQ(status, Task::Status::TS_DONE);
}

TEST_F(TaskExecutorTests, canRunDefaultTaskWithoutWait) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    auto defaultTask = std::make_shared<Task>();
    taskExecutor->startTask(defaultTask);
}

TEST_F(TaskExecutorTests, canRunCustomFunction) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    int i = 0;
    auto customTask = std::make_shared<Task>([&i]() { i++; });
    taskExecutor->startTask(customTask);
    auto status = customTask->wait(-1);
    ASSERT_EQ(status, Task::Status::TS_DONE);
    ASSERT_EQ(i, 1);
}

TEST_F(TaskExecutorTests, canRun2FunctionsOneByOne) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    int i = 0;
    auto task1 = std::make_shared<Task>([&i]() { i += 1; });
    auto task2 = std::make_shared<Task>([&i]() { i *= 2; });
    taskExecutor->startTask(task1);
    taskExecutor->startTask(task2);

    auto status = task1->wait(-1);
    ASSERT_EQ(status, Task::Status::TS_DONE);
    status = task2->wait(-1);
    ASSERT_EQ(status, Task::Status::TS_DONE);

    ASSERT_EQ(i, 2);
}

TEST_F(TaskExecutorTests, returnFalseIfRunTaskWhichIsRunning) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    auto task = std::make_shared<Task>([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    });
    ASSERT_TRUE(taskExecutor->startTask(task));
    ASSERT_FALSE(taskExecutor->startTask(task));
}

TEST_F(TaskExecutorTests, canRun2FunctionsOneByOneWithoutWait) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    auto task1 = std::make_shared<Task>([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    });
    auto task2 = std::make_shared<Task>([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    });
    taskExecutor->startTask(task1);
    taskExecutor->startTask(task2);
}

TEST_F(TaskExecutorTests, canRunMultipleTasksWithExceptionInside) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    std::vector<Task::Ptr> tasks;

    for (int i = 0; i < MAX_NUMBER_OF_TASKS_IN_QUEUE; i++) {
        tasks.push_back(std::make_shared<Task>([]() { throw std::bad_alloc(); }));
    }

    for (auto &task:tasks) {
        taskExecutor->startTask(task);
    }

    for (auto &task:tasks) {
        task->wait(-1);
        ASSERT_TRUE(Task::Status::TS_ERROR == task->getStatus());
        EXPECT_THROW(task->checkException(), std::bad_alloc);
    }
}

// TODO: CVS-11695
TEST_F(TaskExecutorTests, DISABLED_canRunMultipleTasksFromMultipleThreads) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    int sharedVar = 0;
    int THREAD_NUMBER = MAX_NUMBER_OF_TASKS_IN_QUEUE;
    int NUM_INTERNAL_ITERATIONS = 5000;
    std::vector<MetaThread::Ptr> metaThreads;
    std::vector<Task::Ptr> tasks;
    for (int i = 0; i < THREAD_NUMBER; i++) {
        tasks.push_back(
                std::make_shared<Task>([&]() { for (int k = 0; k < NUM_INTERNAL_ITERATIONS; k++) sharedVar++; }));
        metaThreads.push_back(make_shared<MetaThread>([=]() {
            taskExecutor->startTask(tasks.back());
        }));
    }
    for (auto &metaThread : metaThreads) metaThread->waitUntilThreadFinished();

    for (auto &task : tasks) task->wait(-1);

    for (auto &metaThread : metaThreads) metaThread->join();
    for (auto &metaThread : metaThreads) ASSERT_FALSE(metaThread->exceptionWasThrown);
    for (auto &task : tasks) ASSERT_EQ(Task::Status::TS_DONE, task->getStatus());
    ASSERT_EQ(sharedVar, THREAD_NUMBER * NUM_INTERNAL_ITERATIONS);
}

TEST_F(TaskExecutorTests, executorNotReleasedUntilTasksAreDone) {
    std::mutex mutex_block_emulation;
    std::condition_variable cv_block_emulation;
    std::vector<Task::Ptr> tasks;
    bool isBlocked = true;
    int sharedVar = 0;
    for (int i = 0; i < MAX_NUMBER_OF_TASKS_IN_QUEUE; i++) {
        tasks.push_back(std::make_shared<Task>(
                [&]() {
                    // intentionally block task for launching tasks after calling dtor for TaskExecutor
                    std::unique_lock<std::mutex> lock(mutex_block_emulation);
                    cv_block_emulation.wait(lock, [&isBlocked]() { return isBlocked; });
                    sharedVar++;
                })
        );
    }
    {
        auto taskExecutor = std::make_shared<TaskExecutor>();
        for (auto &task : tasks) {
            taskExecutor->startTask(task);
        }
    }
    // time to call dtor for taskExecutor and unlock tasks
    isBlocked = false;
    for (auto &task : tasks) {
        cv_block_emulation.notify_all();
        task->wait(-1);
    }
    // all tasks should be called despite calling dtor for TaskExecutor
    ASSERT_EQ(sharedVar, MAX_NUMBER_OF_TASKS_IN_QUEUE);
}

// TODO: CVS-11695
TEST_F(TaskExecutorTests, DISABLED_startAsyncIsNotBlockedByAnotherTask) {
    std::mutex mutex_block_emulation;
    std::condition_variable cv_block_emulation;
    std::mutex mutex_task_started;
    std::condition_variable cv_task_started;
    bool isStarted = false;
    bool isBlocked = true;
    auto taskExecutor = std::make_shared<TaskExecutor>();

    auto task1 = std::make_shared<Task>([&]() {
        isStarted = true;
        cv_task_started.notify_all();
        // intentionally block task for test purpose
        std::unique_lock<std::mutex> lock(mutex_block_emulation);
        cv_block_emulation.wait(lock, [&isBlocked]() { return !isBlocked; });
    });

    auto task2 = std::make_shared<Task>([&]() {
        std::unique_lock<std::mutex> lock(mutex_task_started);
        cv_task_started.wait(lock, [&isStarted]() { return isStarted; });
    });

    taskExecutor->startTask(task1);
    taskExecutor->startTask(task2);

    isBlocked = false;
    cv_block_emulation.notify_all();
}

TEST_F(TaskExecutorTests, callWaitOnTaskDtorIfProcessed) {
    std::mutex mutex_block_emulation;
    std::condition_variable cv_block_emulation;
    bool isBlocked = true;

    Task::Ptr task = make_shared<Task>([&]() {
        // intentionally block task for calling dtor during launching task
        std::unique_lock<std::mutex> lock(mutex_block_emulation);
        cv_block_emulation.wait(lock, [&isBlocked]() { return !isBlocked; });
    });

    auto taskExecutor = std::make_shared<TaskExecutor>();
    taskExecutor->startTask(task);
    task = nullptr;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    isBlocked = false;
    cv_block_emulation.notify_all();
}
