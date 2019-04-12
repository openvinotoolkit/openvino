// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <thread>

#include <ie_common.h>
#include <details/ie_exception.hpp>
#include <cpp_interfaces/ie_task.hpp>
#include <cpp_interfaces/ie_task_with_stages.hpp>
#include <cpp_interfaces/ie_task_synchronizer.hpp>
#include <cpp_interfaces/ie_task_with_stages.hpp>
#include "task_tests_utils.hpp"


using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class TaskWithStagesTests : public ::testing::Test {
protected:
    StagedTask::Ptr _task;
    int _testVar = -1;

    virtual void SetUp() {
        ASSERT_NO_THROW(_task = std::make_shared<StagedTask>(
                [this]() { if (_task->getStage()) _testVar++; else _testVar--; },
                1));
    }
};

TEST_F(TaskWithStagesTests, canCreateTask) {
    ASSERT_EQ(Task::TS_INITIAL, _task->getStatus());
    ASSERT_EQ(1, _task->getStage());
    ASSERT_EQ(-1, _testVar);
}

TEST_F(TaskWithStagesTests, runNoThrowMakeTaskPostponeWithStages) {
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    ASSERT_EQ(1, _task->getStage());
    ASSERT_EQ(0, _testVar);
    ASSERT_EQ(Task::TS_POSTPONED, _task->getStatus());
}

TEST_F(TaskWithStagesTests, stageDoneReducesStages) {
    ASSERT_EQ(1, _task->getStage());
    ASSERT_NO_THROW(_task->stageDone());
    ASSERT_EQ(0, _task->getStage());
    ASSERT_EQ(-1, _testVar);
}

TEST_F(TaskWithStagesTests, runNoThrowMakesTaskDoneAfterCallStageDone) {
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    ASSERT_EQ(0, _testVar);
    ASSERT_NO_THROW(_task->stageDone());
    ASSERT_EQ(Task::TS_POSTPONED, _task->getStatus());
    ASSERT_NO_THROW(_task->runNoThrowNoBusyCheck());
    ASSERT_EQ(-1, _testVar);
    ASSERT_EQ(Task::TS_DONE, _task->getStatus());
}

TEST_F(TaskWithStagesTests, canResetStages) {
    ASSERT_NO_THROW(_task->stageDone());
    ASSERT_NO_THROW(_task->resetStages());
    ASSERT_EQ(1, _task->getStage());
    ASSERT_EQ(-1, _testVar);
}

TEST_F(TaskWithStagesTests, throwExceptionIfCalledStageDoneMoreThanStagesTimes) {
    ASSERT_NO_THROW(_task->stageDone());
    ASSERT_THROW(_task->stageDone(), InferenceEngineException);
}
