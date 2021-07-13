// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestWaitTests = BehaviorTestsUtils::InferRequestTests;

TEST_P(InferRequestWaitTests, CorrectOneAsyncInferWithGetInOutWithInfWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(InferRequestWaitTests, canStartAsyncInferWithGetInOutWithStatusOnlyWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY);
    ASSERT_TRUE(sts == InferenceEngine::StatusCode::OK || sts == InferenceEngine::StatusCode::RESULT_NOT_READY);
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(InferRequestWaitTests, FailedAsyncInferWithNegativeTimeForWait) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    req.Infer();
    req.StartAsync();
    ASSERT_THROW(req.Wait(-2), InferenceEngine::Exception);
}

TEST_P(InferRequestWaitTests, canWaitWithotStartAsync) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY));
    ASSERT_NO_THROW(req.Wait(1));
}

TEST_P(InferRequestWaitTests, returnDeviceBusyOnSetBlobAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto&& config = configuration;
    auto itConfig = config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
    if (itConfig != config.end()) {
        if (itConfig->second != "CPU_THROUGHPUT_AUTO") {
            if (std::stoi(itConfig->second) == 0) {
                GTEST_SKIP() << "Not applicable with disabled streams";
            }
        }
    }

    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::StatusCode sts;
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY);
    ASSERT_EQ(InferenceEngine::StatusCode::INFER_NOT_STARTED, sts);
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    try {
        req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    } catch (const std::exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY);
    ASSERT_TRUE(sts == InferenceEngine::StatusCode::OK || sts == InferenceEngine::StatusCode::RESULT_NOT_READY);
}

TEST_P(InferRequestWaitTests, returnDeviceBusyOnGetBlobAfterAsyncInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto outputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    InferenceEngine::StatusCode sts;
    req.StartAsync();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob));
}

} // namespace BehaviorTestsDefinitions
