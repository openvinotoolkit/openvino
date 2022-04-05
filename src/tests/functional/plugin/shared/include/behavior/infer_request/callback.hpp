// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "shared_test_classes/subgraph/basic_lstm.hpp"
#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {

using InferRequestCallbackTests = BehaviorTestsUtils::InferRequestTests;

TEST_P(InferRequestCallbackTests, canCallAsyncWithCompletionCallback) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    bool isCalled = false;
    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest r, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::InferRequest request, InferenceEngine::StatusCode status) {
                ASSERT_TRUE(req == request); //the callback is called on the same impl of the request
                // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
                ASSERT_EQ(InferenceEngine::StatusCode::OK, status);
                isCalled = true;
            });

    req.StartAsync();
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);

    ASSERT_EQ(InferenceEngine::StatusCode::OK, waitStatus);
    ASSERT_TRUE(isCalled);
}

TEST_P(InferRequestCallbackTests, syncInferDoesNotCallCompletionCallback) {
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    bool isCalled = false;
    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::InferRequest request, InferenceEngine::StatusCode status) {
                isCalled = true;
            });
    req.Infer();
    ASSERT_FALSE(isCalled);
}

// test that can wait all callbacks on dtor
TEST_P(InferRequestCallbackTests, canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor) {
    const int NUM_ITER = 10;
    struct TestUserData {
        std::atomic<int> numIter = {0};
        std::promise<InferenceEngine::StatusCode> promise;
    };
    TestUserData data;

    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    req.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
            [&](InferenceEngine::InferRequest request, InferenceEngine::StatusCode status) {
                if (status != InferenceEngine::StatusCode::OK) {
                    data.promise.set_value(status);
                } else {
                    if (data.numIter.fetch_add(1) != NUM_ITER) {
                        request.StartAsync();
                    } else {
                        data.promise.set_value(InferenceEngine::StatusCode::OK);
                    }
                }
            });
    auto future = data.promise.get_future();
    req.StartAsync();
    InferenceEngine::StatusCode waitStatus = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, waitStatus);
    future.wait();
    auto callbackStatus = future.get();
    ASSERT_EQ(InferenceEngine::StatusCode::OK, callbackStatus);
    auto dataNumIter = data.numIter - 1;
    ASSERT_EQ(NUM_ITER, dataNumIter);
}

TEST_P(InferRequestCallbackTests, returnGeneralErrorIfCallbackThrowException) {
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    req.SetCompletionCallback([] {
        IE_THROW(GeneralError);
    });

    ASSERT_NO_THROW(req.StartAsync());
    ASSERT_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY), InferenceEngine::GeneralError);
}

TEST_P(InferRequestCallbackTests, LegacyCastAndSetuserDataGetUserData) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    int userData = 0;
    {
        IE_SUPPRESS_DEPRECATED_START
        InferenceEngine::IInferRequest::Ptr ireq = req;
        ASSERT_EQ(InferenceEngine::OK, ireq->SetUserData(static_cast<void*>(&userData), nullptr));
        ASSERT_EQ(InferenceEngine::OK, ireq->SetCompletionCallback(
                [](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) {
                    void* userDataPtr = nullptr;
                    ASSERT_EQ(InferenceEngine::OK, request->GetUserData(&userDataPtr, nullptr));
                    ASSERT_NE(nullptr, userDataPtr);
                    *static_cast<int*>(userDataPtr) = 42;
                }));
        IE_SUPPRESS_DEPRECATED_END
    }
    req.StartAsync();
    req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(42, userData);
}

TEST_P(InferRequestCallbackTests, ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout) {
    // Create CNNNetwork from ngraph::Function
    // return ngrpah::Function
    // GetNetwork(3000, 380) make inference around 20ms on GNA SW
    // so increases chances for getting RESULT_NOT_READY
    function = SubgraphTestsDefinitions::Basic_LSTM_S::GetNetwork(300, 38);
    cnnNet = InferenceEngine::CNNNetwork(function);
    // Load CNNNetwork to target plugins
    execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::StatusCode sts = InferenceEngine::StatusCode::OK;
    std::promise<std::chrono::system_clock::time_point> callbackTimeStamp;
    auto callbackTimeStampFuture = callbackTimeStamp.get_future();
    // add a callback to the request and capture the timestamp
    req.SetCompletionCallback([&]() {
        callbackTimeStamp.set_value(std::chrono::system_clock::now());
    });
    req.StartAsync();
    ASSERT_NO_THROW(sts = req.Wait(InferenceEngine::InferRequest::WaitMode::STATUS_ONLY));
    // get timestamp taken AFTER return from the Wait(STATUS_ONLY)
    const auto afterWaitTimeStamp = std::chrono::system_clock::now();
    // IF the callback timestamp is larger than the afterWaitTimeStamp
    // then we should observe RESULT_NOT_READY
    if (afterWaitTimeStamp < callbackTimeStampFuture.get()) {
        ASSERT_TRUE(sts == InferenceEngine::StatusCode::RESULT_NOT_READY);
    }
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
}

TEST_P(InferRequestCallbackTests, ImplDoseNotCopyCallback) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::CNNNetwork cnnNet(function);
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    auto req = execNet.CreateInferRequest();
    {
        auto somePtr = std::make_shared<int>(42);
        req.SetCompletionCallback([somePtr] {
            ASSERT_EQ(1, somePtr.use_count());
        });
    }

    ASSERT_NO_THROW(req.StartAsync());
    ASSERT_NO_THROW(req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY));
}

}  // namespace BehaviorTestsDefinitions