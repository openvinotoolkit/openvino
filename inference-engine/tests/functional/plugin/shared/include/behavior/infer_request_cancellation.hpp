// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <future>
#include "ie_extension.h"
#include <condition_variable>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <functional_test_utils/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
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
    bool cancelled = false;
    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::InferRequest request, InferenceEngine::StatusCode status) {
                if (targetDevice == CommonTestUtils::DEVICE_CPU) {
                    cancelled = (status == InferenceEngine::StatusCode::INFER_CANCELLED);
                }
            });

    req.StartAsync();
    InferenceEngine::StatusCode cancelStatus = req.Cancel();
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    if (targetDevice == CommonTestUtils::DEVICE_CPU) {
        ASSERT_EQ(true, cancelStatus == InferenceEngine::StatusCode::OK ||
                        cancelStatus == InferenceEngine::StatusCode::INFER_NOT_STARTED);
        if (cancelStatus == InferenceEngine::StatusCode::OK) {
            ASSERT_EQ(true, cancelled);
            ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::INFER_CANCELLED), waitStatus);
        } else {
            ASSERT_EQ(false, cancelled);
            ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), waitStatus);
        }
    } else {
        ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::NOT_IMPLEMENTED), cancelStatus);
        ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), waitStatus);
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

    req.StartAsync();
    req.Cancel();
    req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    req.StartAsync();
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), waitStatus);
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

    InferenceEngine::StatusCode cancelStatus = req.Cancel();

    if (targetDevice == CommonTestUtils::DEVICE_CPU) {
        ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::INFER_NOT_STARTED), cancelStatus);
    } else {
        ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::NOT_IMPLEMENTED), cancelStatus);
    }
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

    const auto statusOnly = InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY;
    while (req.Wait(statusOnly) == InferenceEngine::StatusCode::INFER_NOT_STARTED) {
    }

    InferenceEngine::StatusCode cancelStatus = req.Cancel();
    InferenceEngine::StatusCode inferStatus = InferenceEngine::StatusCode::OK;

    try {
        infer.get();
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        inferStatus = ex.getStatus();
    }

    if (targetDevice == CommonTestUtils::DEVICE_CPU) {
        if (cancelStatus == InferenceEngine::StatusCode::OK) {
            ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::INFER_CANCELLED), inferStatus);
        } else {
            ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), inferStatus);
        }
    } else {
        ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::NOT_IMPLEMENTED), cancelStatus);
        ASSERT_EQ(static_cast<int>(InferenceEngine::StatusCode::OK), inferStatus);
    }
}
}  // namespace BehaviorTestsDefinitions
