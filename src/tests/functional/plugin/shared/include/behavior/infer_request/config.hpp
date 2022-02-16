// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "threading/ie_executor_manager.hpp"

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
using namespace CommonTestUtils;

typedef std::tuple<
        size_t,                             // Stream executor number
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> InferRequestParams;

class InferRequestConfigTest : public testing::WithParamInterface<InferRequestParams>,
                               public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(streamExecutorNumber, targetDevice, configuration) = this->GetParam();
        // Create CNNNetwork from ngrpah::Function
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(targetDevice);
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        std::string targetDevice;
        size_t streamExecutorNumber;
        std::map<std::string, std::string> configuration;
        std::tie(streamExecutorNumber, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "streamExecutorNumber=" << targetDevice << "_";
        if (!configuration.empty()) {
            result << "config=" << configuration;
        }
        return result.str();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
    }

protected:
    InferenceEngine::CNNNetwork cnnNet;
    InferenceEngine::ExecutableNetwork execNet;
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    size_t streamExecutorNumber;

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
        auto req = execNet.CreateInferRequest();
        return req;
    }
};

TEST_P(InferRequestConfigTest, canSetExclusiveAsyncRequests) {
    ASSERT_EQ(0ul, InferenceEngine::executorManager()->getExecutorsNumber());
    ASSERT_NO_THROW(createInferRequestWithConfig());
    if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_EQ(streamExecutorNumber, InferenceEngine::executorManager()->getExecutorsNumber());
    }
}

TEST_P(InferRequestConfigTest, withoutExclusiveAsyncRequests) {
    ASSERT_EQ(0u, InferenceEngine::executorManager()->getExecutorsNumber());
    ASSERT_NO_THROW(createInferRequestWithConfig());
    if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_EQ(streamExecutorNumber, InferenceEngine::executorManager()->getExecutorsNumber());
    }
}

TEST_P(InferRequestConfigTest, ReusableCPUStreamsExecutor) {
    ASSERT_EQ(0u, InferenceEngine::executorManager()->getExecutorsNumber());
    ASSERT_EQ(0u, InferenceEngine::executorManager()->getIdleCPUStreamsExecutorsNumber());

    {
        // Load config
        std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(NO)}};
        config.insert(configuration.begin(), configuration.end());
        if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
            ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
        }
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, targetDevice, config);
        execNet.CreateInferRequest();
        if ((targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
            (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
            ASSERT_EQ(1u, InferenceEngine::executorManager()->getExecutorsNumber());
            ASSERT_EQ(0u, InferenceEngine::executorManager()->getIdleCPUStreamsExecutorsNumber());
        } else if ((targetDevice == CommonTestUtils::DEVICE_AUTO) ||
                   (targetDevice == CommonTestUtils::DEVICE_MULTI)) {
        } else {
            ASSERT_EQ(0u, InferenceEngine::executorManager()->getExecutorsNumber());
            ASSERT_GE(2u, InferenceEngine::executorManager()->getIdleCPUStreamsExecutorsNumber());
        }
    }
}
}  // namespace BehaviorTestsDefinitions