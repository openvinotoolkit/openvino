// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestMultithreadingTests = BehaviorTestsUtils::InferRequestTests;

TEST_P(InferRequestMultithreadingTests, canRun3SyncRequestsConsistentlyFromThreads) {
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();


    auto f1 = std::async(std::launch::async, [&] { req1.Infer(); });
    auto f2 = std::async(std::launch::async, [&] { req2.Infer(); });
    auto f3 = std::async(std::launch::async, [&] { req3.Infer(); });

    ASSERT_NO_THROW(f1.get());
    ASSERT_NO_THROW(f2.get());
    ASSERT_NO_THROW(f3.get());
}

TEST_P(InferRequestMultithreadingTests, canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait) {
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();
    InferenceEngine::StatusCode sts1, sts2, sts3;

    req1.Infer();
    req2.Infer();
    req3.Infer();

    std::thread t1([&] {
        req1.StartAsync();
        sts1 = req1.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    });
    std::thread t2([&] {
        req2.StartAsync();
        sts2 = req2.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    });
    std::thread t3([&] {
        req3.StartAsync();
        sts3 = req3.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    });

    t1.join();
    t2.join();
    t3.join();

    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts1);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts2);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts3);
}

TEST_P(InferRequestMultithreadingTests, canRun3AsyncRequestsConsistentlyWithWait) {
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();

    req1.StartAsync();
    ASSERT_NO_THROW(req1.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));

    req2.Infer();
    ASSERT_NO_THROW(req2.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));

    req3.Infer();
    ASSERT_NO_THROW(req3.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
}
} // namespace BehaviorTestsDefinitions
