// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <threading/ie_executor_manager.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

TEST(ExecutorManagerTests, canCreateSingleExecutorManager) {
    auto execManager = executorManager();
    auto executorManager1 = executorManager();
    auto executorManager2 = executorManager();

    ASSERT_EQ(executorManager1, executorManager2);
}

TEST(ExecutorManagerTests, createDifferentExecutorsForDifferentDevices) {
    auto execManager = executorManager();
    auto executor1 = execManager->getExecutor("CPU");
    auto executor2 = execManager->getExecutor("GPU");

    ASSERT_NE(executor1, executor2);
    ASSERT_EQ(2, execManager->getExecutorsNumber());
}

TEST(ExecutorManagerTests, returnTheSameExecutorForTheSameDevice) {
    auto execManager = executorManager();
    auto executor1 = execManager->getExecutor("CPU");
    auto executor2 = execManager->getExecutor("GPU");

    auto executor = execManager->getExecutor("GPU");

    ASSERT_EQ(executor, executor2);
    ASSERT_EQ(2, execManager->getExecutorsNumber());
}
