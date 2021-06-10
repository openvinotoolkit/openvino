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
#include <vpu/vpu_plugin_config.hpp>
#include <gna/gna_config.hpp>
#include <ie_core.hpp>
#include <threading/ie_executor_manager.hpp>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace BehaviorTestsDefinitions {
// TODO: rename to SetupInferWithConfigTests
using InferConfigTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferConfigTests, canSetExclusiveAsyncRequests) {
    ASSERT_EQ(0ul, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
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
    } else if ((targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
               (targetDevice == CommonTestUtils::DEVICE_KEEMBAY)) {
        ASSERT_EQ(2u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if ((targetDevice == CommonTestUtils::DEVICE_AUTO) || (targetDevice == CommonTestUtils::DEVICE_MULTI)) {
    } else {
        ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    }
}

TEST_P(InferConfigTests, withoutExclusiveAsyncRequests) {
    ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
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

    if ((targetDevice == CommonTestUtils::DEVICE_GNA) || (targetDevice == CommonTestUtils::DEVICE_HDDL)) {
        ASSERT_EQ(0u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else if ((targetDevice == CommonTestUtils::DEVICE_AUTO) || (targetDevice == CommonTestUtils::DEVICE_MULTI)) {
    } else if (targetDevice == CommonTestUtils::DEVICE_MYRIAD) {
        ASSERT_EQ(2u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    } else {
        ASSERT_EQ(1u, InferenceEngine::ExecutorManager::getInstance()->getExecutorsNumber());
    }
}

// TODO: rename to InferWithConfigTests
using InferConfigInTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(InferConfigInTests, CanInferWithConfig) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    auto req = execNet.CreateInferRequest();
    ASSERT_NO_THROW(req.Infer());
}
}  // namespace BehaviorTestsDefinitions