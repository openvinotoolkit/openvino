// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <condition_variable>
#include <future>

#include <ie_core.hpp>
#include <ie_extension.h>

#include "shared_test_classes/base/layer_test_utils.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "base/behavior_test_utils.hpp"
#include "behavior/infer_request_cancellation.hpp"

namespace BehaviorTestsDefinitions {
using CancellationTests = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(CancellationTests, canCancelAsyncRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> largeNetwork = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, 640, 640});
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(largeNetwork);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    req.StartAsync();

    ASSERT_NO_THROW(req.Cancel());
    try {
        req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    } catch (const InferenceEngine::InferCancelled&) {
        SUCCEED();
    }
}

TEST_P(CancellationTests, canResetAfterCancelAsyncRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();

    ASSERT_NO_THROW(req.StartAsync());
    ASSERT_NO_THROW(req.Cancel());
    try {
        req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    } catch (const InferenceEngine::InferCancelled&) {
        SUCCEED();
    }

    ASSERT_NO_THROW(req.StartAsync());
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
}

TEST_P(CancellationTests, canCancelBeforeAsyncRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();

    ASSERT_NO_THROW(req.Cancel());
}

TEST_P(CancellationTests, canCancelInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create function with large input, to have a time to Cancel request
    std::shared_ptr<ngraph::Function> largeNetwork = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, 640, 640});
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(largeNetwork);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();

    auto infer = std::async(std::launch::async, [&req]{ req.Infer(); });

    const auto statusOnly = InferenceEngine::InferRequest::WaitMode::STATUS_ONLY;
    while (req.Wait(statusOnly) == InferenceEngine::StatusCode::INFER_NOT_STARTED) {
    }

    ASSERT_NO_THROW(req.Cancel());
    try {
        infer.get();
    } catch (const InferenceEngine::InferCancelled&) {
        SUCCEED();
    }
}
}  // namespace BehaviorTestsDefinitions
