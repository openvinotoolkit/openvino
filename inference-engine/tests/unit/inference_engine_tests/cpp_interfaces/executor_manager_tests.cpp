// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cpp_interfaces/ie_executor_manager.hpp>
#include <ie_device.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class ExecutorManagerTests : public ::testing::Test {
public:
    ExecutorManagerImpl _manager;
};

TEST_F(ExecutorManagerTests, canCreateSingleExecutorManager) {
    ExecutorManager *executorManager1 = ExecutorManager::getInstance();

    ExecutorManager *executorManager2 = ExecutorManager::getInstance();

    ASSERT_EQ(executorManager1, executorManager2);
}

TEST_F(ExecutorManagerTests, createDifferentExecutorsForDifferentDevices) {
    auto device1 = TargetDeviceInfo::name(TargetDevice::eCPU);
    auto device2 = TargetDeviceInfo::name(TargetDevice::eGPU);

    auto executor1 = _manager.getExecutor(device1);
    auto executor2 = _manager.getExecutor(device2);

    ASSERT_NE(executor1, executor2);
    ASSERT_EQ(2, _manager.getExecutorsNumber());
}

TEST_F(ExecutorManagerTests, returnTheSameExecutorForTheSameDevice) {
    auto device1 = TargetDeviceInfo::name(TargetDevice::eCPU);
    auto device2 = TargetDeviceInfo::name(TargetDevice::eGPU);
    auto executor1 = _manager.getExecutor(device1);
    auto executor2 = _manager.getExecutor(device2);

    auto executor = _manager.getExecutor(device2);

    ASSERT_EQ(executor, executor2);
    ASSERT_EQ(2, _manager.getExecutorsNumber());
}
