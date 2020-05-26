// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>
#include <threading/ie_executor_manager.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "behavior/infer_request_config.hpp"

namespace LayerTestsDefinitions {
    std::string InferConfigTests::getTestCaseName(testing::TestParamInfo<InferConfigParams> obj) {
        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

    void InferConfigTests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void InferConfigTests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

TEST_P(InferConfigTests, canSetExclusiveAsyncRequests) {
    ASSERT_EQ(0ul, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load config
    std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)}};
    config.insert(configuration.begin(), configuration.end());
    if (targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
    targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
    }
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, config);
    execNet.CreateInferRequest();

    if ((targetDevice == CommonTestUtils::DEVICE_HDDL) || (targetDevice == CommonTestUtils::DEVICE_GNA)) {
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if ((targetDevice == CommonTestUtils::DEVICE_FPGA) || (targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
        (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
        ASSERT_EQ(2u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (targetDevice == CommonTestUtils::DEVICE_MULTI) {
    } else {
        ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    }
    function.reset();
    }

TEST_P(InferConfigTests, withoutExclusiveAsyncRequests) {
    ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load config
    std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)}};
    config.insert(configuration.begin(), configuration.end());
    if (targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
    }
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, config);
    execNet.CreateInferRequest();

    if ((targetDevice == CommonTestUtils::DEVICE_GNA) || (targetDevice == CommonTestUtils::DEVICE_HDDL)) {
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (targetDevice == CommonTestUtils::DEVICE_MULTI) {
    } else if (targetDevice == CommonTestUtils::DEVICE_MYRIAD) {
        ASSERT_EQ(2u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else {
        ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    }
    function.reset();
}

    std::string InferConfigInTests::getTestCaseName(testing::TestParamInfo<InferConfigParams> obj) {
        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            result << "configItem=" << configuration.begin()->first << "_" << configuration.begin()->second;
        }
        return result.str();
    }

    void InferConfigInTests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void InferConfigInTests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

TEST_P(InferConfigInTests, CanInferWithConfig) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        // Get Core from cache
        auto ie = PluginCache::get().ie();
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
        auto req = execNet.CreateInferRequest();
        ASSERT_NO_THROW(req.Infer());
        function.reset();
    }

}  // namespace LayerTestsDefinitions