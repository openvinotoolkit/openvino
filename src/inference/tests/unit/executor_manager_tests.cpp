// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/runtime/threading/executor_manager.hpp"

using namespace ::testing;

TEST(ExecutorManagerTests, canCreateSingleExecutorManager) {
    auto executorManager1 = ov::threading::executor_manager();

    auto executorManager2 = ov::threading::executor_manager();

    ASSERT_EQ(executorManager1, executorManager2);
}

TEST(ExecutorManagerTests, createDifferentExecutorsForDifferentDevices) {
    auto executorMgr = ov::threading::executor_manager();
    auto executor1 = executorMgr->get_executor("CPU");
    auto executor2 = executorMgr->get_executor("GPU");

    ASSERT_NE(executor1, executor2);
    ASSERT_EQ(2, executorMgr->get_executors_number());
}

TEST(ExecutorManagerTests, returnTheSameExecutorForTheSameDevice) {
    auto executorMgr = ov::threading::executor_manager();
    auto executor1 = executorMgr->get_executor("CPU");
    auto executor2 = executorMgr->get_executor("GPU");

    auto executor = executorMgr->get_executor("GPU");

    ASSERT_EQ(executor, executor2);
    ASSERT_EQ(2, executorMgr->get_executors_number());
}
