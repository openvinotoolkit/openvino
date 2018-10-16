// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <thread>

#include <ie_common.h>
#include <details/ie_exception.hpp>
#include <cpp_interfaces/ie_task_synchronizer.hpp>
#include <cpp_interfaces/mock_task_synchronizer.hpp>


using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class TaskSynchronizerTests : public ::testing::Test {
protected:
    TaskSynchronizer::Ptr _taskSynchronizer;
    ResponseDesc dsc;

    virtual void TearDown() {
    }

    virtual void SetUp() {
        _taskSynchronizer = std::make_shared<TaskSynchronizer>();
    }
};

TEST_F(TaskSynchronizerTests, canLockAndUnlockRequestSync) {
    _taskSynchronizer->lock();
    ASSERT_EQ(_taskSynchronizer->queueSize(), 1);
    _taskSynchronizer->unlock();
    ASSERT_EQ(_taskSynchronizer->queueSize(), 0);
}

TEST_F(TaskSynchronizerTests, canCallUnlockMoreThanLock) {
    _taskSynchronizer->lock();
    _taskSynchronizer->unlock();
    _taskSynchronizer->unlock();
    ASSERT_EQ(_taskSynchronizer->queueSize(), 0);
}

TEST_F(TaskSynchronizerTests, canUseScopeRequestForMoreThanMaxInQueueTimes) {
    for (int i = 0; i < MAX_NUMBER_OF_TASKS_IN_QUEUE * 10; i++) {
        _taskSynchronizer->lock();
        ASSERT_EQ(_taskSynchronizer->queueSize(), 1);
        _taskSynchronizer->unlock();
        ASSERT_EQ(_taskSynchronizer->queueSize(), 0);
    }
}

TEST_F(TaskSynchronizerTests, canSyncMaxNumThreadsUsingRequestSync) {
    std::vector<std::thread> threads;
    bool achievedMax = false;
    for (int i = 0; i < MAX_NUMBER_OF_TASKS_IN_QUEUE; i++) {
        threads.push_back(std::thread([&]() {
            EXPECT_NO_THROW(_taskSynchronizer->lock());
            // experimental sleep time to achieve maximum of queue
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (_taskSynchronizer->queueSize() == MAX_NUMBER_OF_TASKS_IN_QUEUE) achievedMax = true;
            _taskSynchronizer->unlock();
        }));
    }

    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();

    }
    ASSERT_EQ(achievedMax, true) << "Test error: increase sleep time or disable test";
}

TEST_F(TaskSynchronizerTests, throwExceptionOnLockMoreThanMaxUsingRequestSync) {
    std::string refError = "Failed to add more than " + std::to_string(MAX_NUMBER_OF_TASKS_IN_QUEUE) + " tasks to queue";
    std::string actual;
    std::vector<std::thread> threads;
    bool achievedMax = false;
    for (int i = 0; i < MAX_NUMBER_OF_TASKS_IN_QUEUE + 1; i++) {
        threads.push_back(std::thread([&]() {
            try {
                _taskSynchronizer->lock();
            } catch (InferenceEngine::details::InferenceEngineException iee) {
                actual = iee.what();
                return;
            }
            // experimental sleep time to achieve maximum of queue
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (_taskSynchronizer->queueSize() == MAX_NUMBER_OF_TASKS_IN_QUEUE) achievedMax = true;
            _taskSynchronizer->unlock();
        }));
    }

    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();

    }
    ASSERT_EQ(achievedMax, true)
                                << "Test error: not able to achieve maximum of requests in queue. Increase sleep time or disable test";
    ASSERT_FALSE(actual.empty()) << "Exception wasn't caught: increase sleep time or disable test";
    ASSERT_EQ(actual.substr(0, refError.length()), refError) << "Caught exception does not match expected one";
}

TEST_F(TaskSynchronizerTests, canSyncNThreadsUsingRequestSync) {
    int sharedVar = 0;
    int THREAD_NUMBER = MAX_NUMBER_OF_TASKS_IN_QUEUE;
    int NUM_INTERNAL_ITERATIONS = 5000;
    std::vector<std::thread> threads;

    for (int i = 0; i < THREAD_NUMBER; i++) {
        threads.push_back(std::thread([&]() {
            _taskSynchronizer->lock();
            for (int k = 0; k < NUM_INTERNAL_ITERATIONS; k++) sharedVar++;
            _taskSynchronizer->unlock();
        }));
    }
    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();
    }

    ASSERT_EQ(sharedVar, THREAD_NUMBER * NUM_INTERNAL_ITERATIONS);
}

TEST_F(TaskSynchronizerTests, canSyncNThreadsUsingScopeSync) {
    int sharedVar = 0;
    int THREAD_NUMBER = MAX_NUMBER_OF_TASKS_IN_QUEUE;
    int NUM_INTERNAL_ITERATIONS = 5000;
    std::vector<std::thread> threads;

    for (int i = 0; i < THREAD_NUMBER; i++) {
        threads.push_back(std::thread([&]() {
            ScopedSynchronizer ss(_taskSynchronizer);
            for (int k = 0; k < NUM_INTERNAL_ITERATIONS; k++) sharedVar++;
        }));
    }
    for (auto &thread : threads) {
        if (thread.joinable()) thread.join();
    }

    ASSERT_EQ(sharedVar, THREAD_NUMBER * NUM_INTERNAL_ITERATIONS);
}

TEST_F(TaskSynchronizerTests, callAddToQueueAndWaitOnLock) {
    MockTaskSynchronizerPrivate::Ptr mockTaskSynchronizerPrivate = make_shared<MockTaskSynchronizerPrivate>();

    EXPECT_CALL(*mockTaskSynchronizerPrivate.get(), _addTaskToQueue()).WillOnce(Return(0));
    EXPECT_CALL(*mockTaskSynchronizerPrivate.get(), _waitInQueue(0));

    mockTaskSynchronizerPrivate->lock();
}

TEST_F(TaskSynchronizerTests, callLockOnCreation) {
    MockTaskSynchronizer::Ptr mockTaskSynchronizer = make_shared<MockTaskSynchronizer>();
    TaskSynchronizer::Ptr taskSync = std::dynamic_pointer_cast<TaskSynchronizer>(mockTaskSynchronizer);

    EXPECT_CALL(*mockTaskSynchronizer.get(), lock()).Times(1);
    EXPECT_CALL(*mockTaskSynchronizer.get(), unlock()).Times(1);

    ScopedSynchronizer scopedSynchronizer(taskSync);
}
