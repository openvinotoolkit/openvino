// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <thread>

#include <ie_common.h>
#include <details/ie_exception.hpp>
#include <cpp_interfaces/ie_task.hpp>
#include <cpp_interfaces/ie_task_executor.hpp>
#include <cpp_interfaces/ie_task_with_stages.hpp>
#include <cpp_interfaces/ie_task_synchronizer.hpp>
#include "task_tests_utils.hpp"


using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

enum TaskFlavor {
    BASE_TASK,
    STAGED_TASK,
    BASE_WITH_CALLBACK,
    STAGED_WITH_CALLBACK
};

class TaskCommonTests : public ::testing::Test, public testing::WithParamInterface<TaskFlavor> {
protected:
    Task::Ptr _task;

    Task::Ptr createTask(std::function<void()> function = nullptr, bool forceNull = false) {
        TaskFlavor flavor = GetParam();
        bool condition = function || forceNull;
        Task::Ptr baseTask = condition ? make_shared<Task>(function) : make_shared<Task>();
        Task::Ptr stagedTask = condition ? make_shared<StagedTask>(function, 1) : make_shared<StagedTask>();
        auto executor = make_shared<TaskExecutor>();
        switch (flavor) {
            case BASE_TASK:
                return baseTask;
            case STAGED_TASK:
                return stagedTask;
            default:
                throw logic_error("Specified non-existent flavor of task");
        }
    }
};

TEST_P(TaskCommonTests, canCreateTask) {
    ASSERT_NO_THROW(_task = createTask());
    ASSERT_EQ(_task->getStatus(), Task::TS_INITIAL);
}

TEST_P(TaskCommonTests, canSetBusyStatus) {
    ASSERT_NO_THROW(_task = createTask());
    ASSERT_NO_THROW(_task->occupy());
    ASSERT_EQ(_task->getStatus(), Task::TS_BUSY);
}

TEST_P(TaskCommonTests, firstOccupyReturnTrueSecondFalse) {
    ASSERT_NO_THROW(_task = createTask());
    ASSERT_TRUE(_task->occupy());
    ASSERT_FALSE(_task->occupy());
}

TEST_P(TaskCommonTests, canRunDefaultTask) {
    _task = createTask();
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    ASSERT_EQ(_task->getStatus(), Task::TS_DONE);
}

TEST_P(TaskCommonTests, throwIfFunctionNull) {
    ASSERT_THROW(_task = createTask(nullptr, true), InferenceEngineException);
}

TEST_P(TaskCommonTests, canWaitWithoutRun) {
    _task = createTask();
    ASSERT_NO_THROW(_task->wait(-1));
    ASSERT_EQ(_task->getStatus(), Task::TS_INITIAL);
    ASSERT_NO_THROW(_task->wait(1));
    ASSERT_EQ(_task->getStatus(), Task::TS_INITIAL);
}

TEST_P(TaskCommonTests, canRunTaskFromThread) {
    _task = createTask();

    MetaThread metaThread([=]() {
        _task->runNoThrowNoBusyCheck();
    });

    metaThread.join();
    ASSERT_EQ(Task::Status::TS_DONE, _task->getStatus());
}


TEST_P(TaskCommonTests, canRunTaskFromThreadWithoutWait) {
    _task = createTask([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    });
    std::thread thread([this]() { _task->runNoThrowNoBusyCheck(); });
    if (thread.joinable()) thread.join();
}

TEST_P(TaskCommonTests, waitReturnNotStartedIfTaskWasNotRun) {
    _task = createTask();
    Task::Status status = _task->wait(1);
    ASSERT_EQ(status, Task::Status::TS_INITIAL);
}

TEST_P(TaskCommonTests, canCatchIEException) {
    _task = createTask([]() { THROW_IE_EXCEPTION; });
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    Task::Status status = _task->getStatus();
    ASSERT_EQ(status, Task::Status::TS_ERROR);
    EXPECT_THROW(_task->checkException(), InferenceEngineException);
}

TEST_P(TaskCommonTests, waitReturnErrorIfException) {
    _task = createTask([]() { THROW_IE_EXCEPTION; });
    ASSERT_NO_THROW(_task->occupy());
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    Task::Status status = _task->wait(-1);
    ASSERT_EQ(status, Task::Status::TS_ERROR);
    EXPECT_THROW(_task->checkException(), InferenceEngineException);
}

TEST_P(TaskCommonTests, canCatchStdException) {
    _task = createTask([]() { throw std::bad_alloc(); });
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    Task::Status status = _task->getStatus();
    ASSERT_EQ(status, Task::Status::TS_ERROR);
    EXPECT_THROW(_task->checkException(), std::bad_alloc);
}

TEST_P(TaskCommonTests, canCleanExceptionPtr) {
    bool throwException = true;
    _task = createTask([&throwException]() { if (throwException) throw std::bad_alloc(); else return; });
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    EXPECT_THROW(_task->checkException(), std::bad_alloc);
    throwException = false;
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    EXPECT_NO_THROW(_task->checkException());
}

std::string getTestCaseName(testing::TestParamInfo<TaskFlavor> obj) {
#define CASE(x) case x: return #x;
    switch (obj.param) {
        CASE(BASE_TASK);
        CASE(STAGED_TASK);
        CASE(BASE_WITH_CALLBACK);
        CASE(STAGED_WITH_CALLBACK);
        default :
            return "EMPTY";
#undef CASE
    }
}

INSTANTIATE_TEST_CASE_P(Task, TaskCommonTests,
                        ::testing::ValuesIn(std::vector<TaskFlavor>{BASE_TASK, STAGED_TASK}), getTestCaseName);
