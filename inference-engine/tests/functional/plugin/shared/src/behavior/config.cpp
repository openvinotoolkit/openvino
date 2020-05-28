// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "ie_common.h"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include <threading/ie_executor_manager.hpp>
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "behavior/config.hpp"


namespace LayerTestsDefinitions {
std::string CorrectConfigTests::getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
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

void CorrectConfigTests::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
    function = ngraph::builder::subgraph::makeConvPoolRelu();
}

void CorrectConfigTests::TearDown() {
    if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
        PluginCache::get().reset();
    }
}

// Setting empty config doesn't throw
TEST_P(CorrectConfigTests, SetEmptyConfig) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    std::map<std::string, std::string> config;
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
    function.reset();
}

// Setting correct config doesn't throw
TEST_P(CorrectConfigTests, SetCorrectConfig) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
    function.reset();
}

    std::string IncorrectConfigTests::getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
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

    void IncorrectConfigTests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void IncorrectConfigTests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

TEST_P(IncorrectConfigTests, SetConfigWithIncorrectKey) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    if (targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        ASSERT_THROW(ie->SetConfig(configuration, targetDevice),
                     InferenceEngine::details::InferenceEngineException);
    } else {
        ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
    }
    function.reset();
}

TEST_P(IncorrectConfigTests, canNotLoadNetworkWithIncorrectConfig) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    ASSERT_THROW(auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration),
                         InferenceEngine::details::InferenceEngineException);
    function.reset();
    }

    std::string IncorrectConfigAPITests::getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
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

    void IncorrectConfigAPITests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void IncorrectConfigAPITests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

TEST_P(IncorrectConfigAPITests, SetConfigWithNoExistingKey) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    if (targetDevice.find(CommonTestUtils::DEVICE_GNA) != std::string::npos) {
        ASSERT_THROW(ie->SetConfig(configuration, targetDevice), InferenceEngine::NotFound);
    } else {
        try {
            ie->SetConfig(configuration, targetDevice);
        } catch (InferenceEngine::details::InferenceEngineException ex) {}
    }
    function.reset();
    }


    std::string CorrectConfigAPITests::getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
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

    void CorrectConfigAPITests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void CorrectConfigAPITests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

TEST_P(CorrectConfigAPITests, canSetExclusiveAsyncRequests) {
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
        ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
    }
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    execNet.CreateInferRequest();

    if ((targetDevice == CommonTestUtils::DEVICE_HDDL) || (targetDevice == CommonTestUtils::DEVICE_GNA) ||
        (targetDevice == CommonTestUtils::DEVICE_CPU) || (targetDevice == CommonTestUtils::DEVICE_GPU)) {
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if ((targetDevice == CommonTestUtils::DEVICE_FPGA) ||
        (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
        ASSERT_EQ(2u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (targetDevice == CommonTestUtils::DEVICE_MULTI) {
    } else {
        ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    }

    function.reset();
}

TEST_P(CorrectConfigAPITests, withoutExclusiveAsyncRequests) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load config
    std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(NO)}};
    config.insert(configuration.begin(), configuration.end());
    if (targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
    }
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
        execNet.CreateInferRequest();

    if ((targetDevice == CommonTestUtils::DEVICE_FPGA) || (targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
            (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
        ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (targetDevice == CommonTestUtils::DEVICE_MULTI) {
    } else {
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    }
    function.reset();
}

TEST_P(CorrectConfigAPITests, reusableCPUStreamsExecutor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());

    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
    auto ie = PluginCache::get().ie();
    {
        // Load config
        std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(NO)}};
        config.insert(configuration.begin(), configuration.end());
        if (targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
            ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
        }
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
        execNet.CreateInferRequest();

        if ((targetDevice == CommonTestUtils::DEVICE_FPGA) || (targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
            (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
            ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
            ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
        } else if (targetDevice == CommonTestUtils::DEVICE_MULTI) {
        } else {
            ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
            ASSERT_GE(2u, InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
            }
    }
    if (targetDevice == CommonTestUtils::DEVICE_CPU) {
        ASSERT_NE(0u, InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
        ASSERT_NO_THROW(ie->UnregisterPlugin("CPU"));
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
    }
    function.reset();
}
}  // namespace LayerTestsDefinitions