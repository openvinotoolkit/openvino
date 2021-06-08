// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "ie_extension.h"
#include <condition_variable>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_plugin_config.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <gna/gna_config.hpp>
#include <ie_core.hpp>
#include "ie_common.h"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include <threading/ie_executor_manager.hpp>
#include <base/behavior_test_utils.hpp>
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace BehaviorTestsDefinitions {
    using CorrectConfigTests = BehaviorTestsUtils::BehaviorTestsBasic;
    // Setting empty config doesn't throw
    TEST_P(CorrectConfigTests, SetEmptyConfig) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        std::map<std::string, std::string> config;
        ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
    }

    // Setting correct config doesn't throw
    TEST_P(CorrectConfigTests, SetCorrectConfig) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
    }

    using IncorrectConfigTests = BehaviorTestsUtils::BehaviorTestsBasic;

    TEST_P(IncorrectConfigTests, SetConfigWithIncorrectKey) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        if (targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
            ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
            ASSERT_THROW(ie->SetConfig(configuration, targetDevice),
                         InferenceEngine::Exception);
        } else {
            ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
            ASSERT_NO_THROW(ie->SetConfig(configuration, targetDevice));
        }
    }

    TEST_P(IncorrectConfigTests, canNotLoadNetworkWithIncorrectConfig) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) != std::string::npos) {
            GTEST_SKIP();
        } else {
            ASSERT_THROW(auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration),
                         InferenceEngine::Exception);
        }
    }

    using IncorrectConfigAPITests = BehaviorTestsUtils::BehaviorTestsBasic;

    TEST_P(IncorrectConfigAPITests, SetConfigWithNoExistingKey) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        ASSERT_NO_THROW(ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        if (targetDevice.find(CommonTestUtils::DEVICE_GNA) != std::string::npos) {
            ASSERT_THROW(ie->SetConfig(configuration, targetDevice), InferenceEngine::NotFound);
        } else {
            try {
                ie->SetConfig(configuration, targetDevice);
            } catch (InferenceEngine::Exception &) {}
        }
    }

    using CorrectConfigAPITests = BehaviorTestsUtils::BehaviorTestsBasic;

    TEST_P(CorrectConfigAPITests, canSetExclusiveAsyncRequests) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        // Load config
        std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)}};
        config.insert(configuration.begin(), configuration.end());
        if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
            ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
        }
        // Load CNNNetwork to target plugins
        if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos) {
            auto execNet = ie->LoadNetwork(cnnNet, targetDevice, config);
            execNet.CreateInferRequest();
        }

        if ((targetDevice == CommonTestUtils::DEVICE_HDDL) || (targetDevice == CommonTestUtils::DEVICE_GNA)) {
            ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
        } else if ((targetDevice == CommonTestUtils::DEVICE_KEEMBAY) ||
                   (targetDevice == CommonTestUtils::DEVICE_MYRIAD)) {
            ASSERT_EQ(2u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
        } else if ((targetDevice == CommonTestUtils::DEVICE_MULTI) ||
                   (targetDevice == CommonTestUtils::DEVICE_AUTO)) {
        } else {
            ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
        }
    }

    TEST_P(CorrectConfigAPITests, withoutExclusiveAsyncRequests) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        // Load config
        std::map<std::string, std::string> config = {{CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(NO)}};
        config.insert(configuration.begin(), configuration.end());
        if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
            targetDevice.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
            ASSERT_NO_THROW(ie->SetConfig(config, targetDevice));
        }
        // Load CNNNetwork to target plugins
        if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos) {
            auto execNet = ie->LoadNetwork(cnnNet, targetDevice, config);
            execNet.CreateInferRequest();
        }

        if ((targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
            (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
            ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
        } else if ((targetDevice == CommonTestUtils::DEVICE_AUTO) ||
                   (targetDevice == CommonTestUtils::DEVICE_MULTI)) {
        } else {
            ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
        }
    }

    TEST_P(CorrectConfigAPITests, reusableCPUStreamsExecutor) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());

        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
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
            if (targetDevice.find(CommonTestUtils::DEVICE_AUTO) == std::string::npos) {
                auto execNet = ie->LoadNetwork(cnnNet, targetDevice, config);
                execNet.CreateInferRequest();
            }

            if ((targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
                (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
                ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
                ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
            } else if ((targetDevice == CommonTestUtils::DEVICE_AUTO) ||
                       (targetDevice == CommonTestUtils::DEVICE_MULTI)) {
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
    }
}  // namespace BehaviorTestsDefinitions