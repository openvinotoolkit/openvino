// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <threading/ie_executor_manager.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

TEST(ExecutorManagerTests, canCreateSingleExecutorManager) {
    auto executorManager1 = executorManager();

    auto executorManager2 = executorManager();

    ASSERT_EQ(executorManager1, executorManager2);
}

TEST(ExecutorManagerTests, createDifferentExecutorsForDifferentDevices) {
    auto executor1 = executorManager()->getExecutor("CPU");
    auto executor2 = executorManager()->getExecutor("GPU");

    ASSERT_NE(executor1, executor2);
    ASSERT_EQ(2, executorManager()->getExecutorsNumber());
}

TEST(ExecutorManagerTests, returnTheSameExecutorForTheSameDevice) {
    auto executor1 = executorManager()->getExecutor("CPU");
    auto executor2 = executorManager()->getExecutor("GPU");

    auto executor = executorManager()->getExecutor("GPU");

    ASSERT_EQ(executor, executor2);
    ASSERT_EQ(2, executorManager()->getExecutorsNumber());
}
