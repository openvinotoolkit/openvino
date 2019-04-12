// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <thread>

#include <ie_common.h>
#include <details/ie_exception.hpp>
#include <cpp_interfaces/ie_task.hpp>
#include <cpp_interfaces/ie_task_synchronizer.hpp>
#include "task_tests_utils.hpp"


using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;


class TaskTests : public ::testing::Test {
protected:
    Task::Ptr _task = std::make_shared<Task>();
};

TEST_F(TaskTests, canRunWithTaskSync) {
    TaskSynchronizer::Ptr taskSynchronizer = std::make_shared<TaskSynchronizer>();
    ASSERT_NO_THROW(_task->runWithSynchronizer(taskSynchronizer));
    ASSERT_EQ(_task->getStatus(), Task::TS_DONE);
}

TEST_F(TaskTests, canRunWithTaskSyncAndWait) {
    TaskSynchronizer::Ptr taskSynchronizer = std::make_shared<TaskSynchronizer>();
    ASSERT_NO_THROW(_task->runWithSynchronizer(taskSynchronizer));
    Task::Status status = _task->wait(-1);
    ASSERT_EQ(status, Task::TS_DONE);
}

// TODO: CVS-11695
TEST_F(TaskTests, DISABLED_returnBusyStatusWhenStartTaskWhichIsRunning) {
    TaskSynchronizer::Ptr taskSynchronizer = std::make_shared<TaskSynchronizer>();
    std::vector<Task::Status> statuses;
    std::vector<MetaThread::Ptr> metaThreads;
    // otherwise push_back to the vector won't be thread-safe
    statuses.reserve(MAX_NUMBER_OF_TASKS_IN_QUEUE);
    _task->occupy();

    for (int i = 0; i < MAX_NUMBER_OF_TASKS_IN_QUEUE; i++) {
        metaThreads.push_back(make_shared<MetaThread>([&]() {
            statuses.push_back(_task->runWithSynchronizer(taskSynchronizer));
        }));
    }

    for (auto &metaThread : metaThreads) metaThread->join();
    for (auto &status : statuses) ASSERT_EQ(Task::Status::TS_BUSY, status) << "Start task never return busy status";
}

TEST_F(TaskTests, canSyncNThreadsUsingTaskSync) {
    TaskSynchronizer::Ptr taskSynchronizer = std::make_shared<TaskSynchronizer>();
    int sharedVar = 0;
    size_t THREAD_NUMBER = MAX_NUMBER_OF_TASKS_IN_QUEUE;
    size_t NUM_INTERNAL_ITERATIONS = 5000;
    std::vector<Task::Status> statuses;
    std::vector<MetaThread::Ptr> metaThreads;
    // otherwise push_back to the vector won't be thread-safe
    statuses.reserve(THREAD_NUMBER);

    for (int i = 0; i < THREAD_NUMBER; i++) {
        metaThreads.push_back(make_shared<MetaThread>([&]() {
            auto status = Task([&]() {
                for (int k = 0; k < NUM_INTERNAL_ITERATIONS; k++) sharedVar++;
            }).runWithSynchronizer(taskSynchronizer);
            statuses.push_back(status);
        }));
    }

    for (auto &metaThread : metaThreads) metaThread->join();
    for (auto &status : statuses) ASSERT_NE(Task::Status::TS_BUSY, status);
    ASSERT_EQ(sharedVar, THREAD_NUMBER * NUM_INTERNAL_ITERATIONS);
}
