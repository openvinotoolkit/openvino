// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <inference_engine.hpp>
#include <cpp_interfaces/impl/mock_async_infer_request_thread_safe_internal.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class AsyncInferRequestThreadSafeInternalTests : public ::testing::Test {
protected:
    MockAsyncInferRequestThreadSafeInternal::Ptr testRequest;
    ResponseDesc dsc;

    bool _doesThrowExceptionWithMessage(std::function<void()> func, string refError) {
        std::string whatMessage;
        try {
            func();
        } catch (const InferenceEngineException &iee) {
            whatMessage = iee.what();
        }
        return whatMessage.find(refError) != std::string::npos;
    }

    virtual void SetUp() {
        testRequest = make_shared<MockAsyncInferRequestThreadSafeInternal>();
    }

};

// StartAsync
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnRequestBusyOnStartAsync) {
    testRequest->setRequestBusy();
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->StartAsync(); }, REQUEST_BUSY_str));
}

TEST_F(AsyncInferRequestThreadSafeInternalTests, canResetBusyStatusIfStartAsyncTaskFails) {
    EXPECT_CALL(*testRequest.get(), StartAsync_ThreadUnsafe()).Times(2)
            .WillOnce(Throw(InferenceEngineException(__FILE__, __LINE__) << "compare"))
            .WillOnce(Return());

    ASSERT_TRUE(_doesThrowExceptionWithMessage([&]() { testRequest->StartAsync(); }, "compare"));
    ASSERT_NO_THROW(testRequest->StartAsync());
}

TEST_F(AsyncInferRequestThreadSafeInternalTests, deviceBusyAfterStartAsync) {
    EXPECT_CALL(*testRequest.get(), StartAsync_ThreadUnsafe()).WillOnce(Return());

    ASSERT_NO_THROW(testRequest->StartAsync());

    ASSERT_TRUE(testRequest->isRequestBusy());
}

// GetUserData
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnRequestBusyOnGetUserData) {
    testRequest->setRequestBusy();
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->GetUserData(nullptr); }, REQUEST_BUSY_str));
}

// SetUserData
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnRequestBusyOnSetUserData) {
    testRequest->setRequestBusy();
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->SetUserData(nullptr); }, REQUEST_BUSY_str));
}

// Wait
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnInferNotStartedOnWait) {
    testRequest->setRequestBusy();
    int64_t ms = 0;
    EXPECT_CALL(*testRequest.get(), Wait(ms)).WillOnce(Return(INFER_NOT_STARTED));

    StatusCode actual = testRequest->Wait(ms);
    ASSERT_EQ(INFER_NOT_STARTED, actual);
}

// Infer
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnRequestBusyOnInfer) {
    testRequest->setRequestBusy();
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->Infer(); }, REQUEST_BUSY_str));
}

TEST_F(AsyncInferRequestThreadSafeInternalTests, canResetBusyStatusIfInferFails) {
    EXPECT_CALL(*testRequest.get(), Infer_ThreadUnsafe()).Times(2)
            .WillOnce(Throw(InferenceEngineException(__FILE__, __LINE__) << "compare"))
            .WillOnce(Return());

    ASSERT_TRUE(_doesThrowExceptionWithMessage([&]() { testRequest->Infer(); }, "compare"));
    ASSERT_NO_THROW(testRequest->Infer());
}

// GetPerformanceCounts
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnRequestBusyOnGetPerformanceCounts) {
    testRequest->setRequestBusy();
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
        testRequest->GetPerformanceCounts(info);
    }, REQUEST_BUSY_str));
}

// GetBlob
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnRequestBusyOnGetBlob) {
    testRequest->setRequestBusy();
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() {
        Blob::Ptr data;
        testRequest->GetBlob(nullptr, data);
    }, REQUEST_BUSY_str));
}

// SetBlob
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnRequestBusyOnSetBlob) {
    testRequest->setRequestBusy();
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->SetBlob(nullptr, nullptr); }, REQUEST_BUSY_str));
}

// SetCompletionCallback
TEST_F(AsyncInferRequestThreadSafeInternalTests, returnRequestBusyOnSetCompletionCallback) {
    testRequest->setRequestBusy();
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->SetCompletionCallback(nullptr); },
                                               REQUEST_BUSY_str));
}
