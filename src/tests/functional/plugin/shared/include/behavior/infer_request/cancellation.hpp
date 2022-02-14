// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {

class InferRequestCancellationTests : public BehaviorTestsUtils::InferRequestTests {
public:
    void SetUp()  override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(targetDevice, configuration) = this->GetParam();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(targetDevice, {1, 3, 640, 640});
        cnnNet = InferenceEngine::CNNNetwork(function);
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    }
};

TEST_P(InferRequestCancellationTests, canCancelAsyncRequest) {
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

TEST_P(InferRequestCancellationTests, canResetAfterCancelAsyncRequest) {
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

TEST_P(InferRequestCancellationTests, canCancelBeforeAsyncRequest) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    ASSERT_NO_THROW(req.Cancel());
}

TEST_P(InferRequestCancellationTests, canCancelInferRequest) {
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
