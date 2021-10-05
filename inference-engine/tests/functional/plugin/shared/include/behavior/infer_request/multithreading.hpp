// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>
#include <memory>

#include "base/behavior_test_utils.hpp"
#include "threading/ie_cpu_streams_executor.hpp"

namespace BehaviorTestsDefinitions {
struct InferRequestMultithreadingTests : BehaviorTestsUtils::InferRequestTests {
    InferRequestMultithreadingTests() : executor{
        InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded({})} {}
    template<typename F>
    std::future<void> async(F&& f) {
        auto p = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
        auto future = p->get_future();
        executor.run([p] () mutable {(*p)();});
        return future;
    }
    InferenceEngine::CPUStreamsExecutor executor;
};

TEST_P(InferRequestMultithreadingTests, canRun3SyncRequestsConsistentlyFromThreads) {
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();


    auto f1 = async([&] { req1.Infer(); });
    auto f2 = async([&] { req2.Infer(); });
    auto f3 = async([&] { req3.Infer(); });

    f1.wait();
    f2.wait();
    f3.wait();

    ASSERT_NO_THROW(f1.get());
    ASSERT_NO_THROW(f2.get());
    ASSERT_NO_THROW(f3.get());
}

TEST_P(InferRequestMultithreadingTests, canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait) {
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();

    req1.Infer();
    req2.Infer();
    req3.Infer();

    auto f1 = async([&] { req1.StartAsync(); });
    auto f2 = async([&] { req2.StartAsync(); });
    auto f3 = async([&] { req3.StartAsync(); });

    f1.wait();
    f2.wait();
    f3.wait();

    ASSERT_NO_THROW(f1.get());
    ASSERT_NO_THROW(f2.get());
    ASSERT_NO_THROW(f3.get());
}

TEST_P(InferRequestMultithreadingTests, canRun3AsyncRequestsConsistentlyWithWait) {
    // Create InferRequest
    auto req1 = execNet.CreateInferRequest();
    auto req2 = execNet.CreateInferRequest();
    auto req3 = execNet.CreateInferRequest();

    req1.StartAsync();
    ASSERT_NO_THROW(req1.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));

    req2.StartAsync();
    ASSERT_NO_THROW(req2.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));

    req3.StartAsync();
    ASSERT_NO_THROW(req3.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
}
} // namespace BehaviorTestsDefinitions
