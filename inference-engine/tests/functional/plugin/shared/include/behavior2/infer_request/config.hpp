// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "threading/ie_executor_manager.hpp"

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
class InferRequestConfigTest : public BehaviorTestsUtils::InferRequestTests {
public:
    void SetUp() override {
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        // Create CNNNetwork from ngrpah::Function
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

protected:
    inline InferenceEngine::InferRequest createInferRequestWithConfig() {
        // Load config
        configuration.insert({CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)});
        if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
            ie->SetConfig(configuration, targetDevice);
        }
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
        return execNet.CreateInferRequest();
    }
};

TEST_P(InferRequestConfigTest, canSetExclusiveAsyncRequests) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ASSERT_EQ(0ul, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    ASSERT_NO_THROW(createInferRequestWithConfig());
    if ((targetDevice == CommonTestUtils::DEVICE_HDDL) || (targetDevice == CommonTestUtils::DEVICE_GNA)) {
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if ((targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
               (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
        ASSERT_EQ(2u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if ((targetDevice == CommonTestUtils::DEVICE_AUTO) || (targetDevice == CommonTestUtils::DEVICE_MULTI)) {
    } else {
        ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    }
}

TEST_P(InferRequestConfigTest, withoutExclusiveAsyncRequests) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    ASSERT_NO_THROW(createInferRequestWithConfig());
    if ((targetDevice == CommonTestUtils::DEVICE_GNA) || (targetDevice == CommonTestUtils::DEVICE_HDDL)) {
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if ((targetDevice == CommonTestUtils::DEVICE_AUTO) || (targetDevice == CommonTestUtils::DEVICE_MULTI) ||
               (targetDevice == CommonTestUtils::DEVICE_HETERO)) {
    } else if (targetDevice == CommonTestUtils::DEVICE_MYRIAD) {
        ASSERT_EQ(2u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else {
        ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    }
}
}  // namespace BehaviorTestsDefinitions