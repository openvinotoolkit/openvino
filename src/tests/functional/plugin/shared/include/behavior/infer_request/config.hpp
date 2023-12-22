// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/executor_manager.hpp"

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {

typedef std::tuple<
        size_t,                             // Stream executor number
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> InferRequestParams;

class InferRequestConfigTest : public testing::WithParamInterface<InferRequestParams>,
                               public BehaviorTestsUtils::IEInferRequestTestBase {
public:
    void SetUp() override {
        std::tie(streamExecutorNumber, target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        // Create CNNNetwork from ngrpah::Function
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
        cnnNet = InferenceEngine::CNNNetwork(function);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
        using namespace ov::test::utils;

        std::string target_device;
        size_t streamExecutorNumber;
        std::map<std::string, std::string> configuration;
        std::tie(streamExecutorNumber, target_device, configuration) = obj.param;
        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        result << "streamExecutorNumber=" << target_device << "_";
        if (!configuration.empty()) {
            result << "config=" << configuration;
        }
        return result.str();
    }

protected:
    InferenceEngine::CNNNetwork cnnNet;
    InferenceEngine::ExecutableNetwork execNet;
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    std::map<std::string, std::string> configuration;
    size_t streamExecutorNumber;

    void set_api_entity() override { api_entity = ov::test::utils::ov_entity::ie_infer_request; }

    inline InferenceEngine::InferRequest createInferRequestWithConfig() {
        // Load config
        configuration.insert({CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)});
        if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
                target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
            ie->SetConfig(configuration, target_device);
        }
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
        auto req = execNet.CreateInferRequest();
        return req;
    }
};

TEST_P(InferRequestConfigTest, canSetExclusiveAsyncRequests) {
    ASSERT_EQ(0ul, ov::threading::executor_manager()->get_executors_number());
    ASSERT_NO_THROW(createInferRequestWithConfig());
    if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
        ASSERT_EQ(streamExecutorNumber, ov::threading::executor_manager()->get_executors_number());
    }
}

TEST_P(InferRequestConfigTest, withoutExclusiveAsyncRequests) {
    ASSERT_EQ(0u, ov::threading::executor_manager()->get_executors_number());
    ASSERT_NO_THROW(createInferRequestWithConfig());
    if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
        target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
        ASSERT_EQ(streamExecutorNumber, ov::threading::executor_manager()->get_executors_number());
    }
}

TEST_P(InferRequestConfigTest, ReusableCPUStreamsExecutor) {
    ASSERT_EQ(0u, ov::threading::executor_manager()->get_executors_number());
    ASSERT_EQ(0u, ov::threading::executor_manager()->get_idle_cpu_streams_executors_number());

    {
        // Load config
        std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(NO)}};
        config.insert(configuration.begin(), configuration.end());
        if (target_device.find(ov::test::utils::DEVICE_AUTO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_MULTI) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_HETERO) == std::string::npos &&
            target_device.find(ov::test::utils::DEVICE_BATCH) == std::string::npos) {
            ASSERT_NO_THROW(ie->SetConfig(config, target_device));
        }
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, target_device, config);
        execNet.CreateInferRequest();
        if (target_device == ov::test::utils::DEVICE_KEEMBAY) {
            ASSERT_EQ(1u, ov::threading::executor_manager()->get_executors_number());
            ASSERT_EQ(0u, ov::threading::executor_manager()->get_idle_cpu_streams_executors_number());
        } else if ((target_device == ov::test::utils::DEVICE_AUTO) ||
                   (target_device == ov::test::utils::DEVICE_MULTI)) {
        } else {
            ASSERT_EQ(0u, ov::threading::executor_manager()->get_executors_number());
            ASSERT_GE(2u, ov::threading::executor_manager()->get_idle_cpu_streams_executors_number());
        }
    }
}
}  // namespace BehaviorTestsDefinitions
