// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include <cpp_interfaces/mock_task_executor.hpp>
#include <mock_iasync_infer_request.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class CallbackManagerTests : public ::testing::Test {
protected:
    CallbackManager::Ptr callbackManager;
    MockTaskExecutor::Ptr mockExecutor;
    IInferRequest::Ptr mockRequest;
    std::function<void(IInferRequest::Ptr, StatusCode code)> mockCallback;

    virtual void TearDown() {
    }

    virtual void SetUp() {
        mockExecutor = make_shared<MockTaskExecutor>();
        callbackManager = make_shared<CallbackManager>(mockExecutor);
        mockCallback = [](IInferRequest::Ptr, StatusCode code) {};
        mockRequest = make_shared<MockIInferRequest>();
    }
};

TEST_F(CallbackManagerTests, disabledByDefault) {
    ASSERT_FALSE(callbackManager->isCallbackEnabled());
}

TEST_F(CallbackManagerTests, disabledIfCallbackNotSet) {
    callbackManager->enableCallback();
    ASSERT_FALSE(callbackManager->isCallbackEnabled());
}

TEST_F(CallbackManagerTests, canStartTask) {
    auto task = make_shared<Task>();
    EXPECT_CALL(*mockExecutor.get(), startTask(task));
    ASSERT_NO_THROW(callbackManager->startTask(task));
}

TEST_F(CallbackManagerTests, canSetCallback) {
    callbackManager->set_callback([](IInferRequest::Ptr, StatusCode code) {});
    ASSERT_TRUE(callbackManager->isCallbackEnabled());
}

TEST_F(CallbackManagerTests, failToRunForEmptyRequest) {
    callbackManager->set_callback([](IInferRequest::Ptr, StatusCode code) {});
    EXPECT_THROW(callbackManager->runCallback(), InferenceEngineException);
}

TEST_F(CallbackManagerTests, callbackIsNotCalledIfWasNotSet) {
    callbackManager->set_publicInterface(mockRequest);
    EXPECT_NO_THROW(callbackManager->runCallback());
}

TEST_F(CallbackManagerTests, canRunCallback) {
    callbackManager->set_publicInterface(mockRequest);
    callbackManager->set_callback([](IInferRequest::Ptr, StatusCode code) {
        throw logic_error("");
    });
    EXPECT_THROW(callbackManager->runCallback(), logic_error);
}

TEST_F(CallbackManagerTests, canDisableCallback) {
    callbackManager->set_callback([](IInferRequest::Ptr, StatusCode code) {});
    callbackManager->disableCallback();
    ASSERT_FALSE(callbackManager->isCallbackEnabled());
}

TEST_F(CallbackManagerTests, canSetException) {
    callbackManager->set_requestException(std::make_exception_ptr(std::logic_error("")));
    callbackManager->set_publicInterface(mockRequest);
    callbackManager->set_callback([](IInferRequest::Ptr, StatusCode code) {});
    EXPECT_THROW(callbackManager->runCallback(), std::logic_error);
}

TEST_F(CallbackManagerTests, canSetStatus) {
    callbackManager->set_requestStatus(GENERAL_ERROR);
    callbackManager->set_publicInterface(mockRequest);
    callbackManager->set_callback([](IInferRequest::Ptr, StatusCode code) {
        ASSERT_EQ(GENERAL_ERROR, code);
    });
}

TEST_F(CallbackManagerTests, initResetStatusAndException) {
    callbackManager->set_requestStatus(GENERAL_ERROR);
    callbackManager->set_requestException(std::make_exception_ptr(std::logic_error("")));
    callbackManager->set_publicInterface(mockRequest);
    callbackManager->set_callback([](IInferRequest::Ptr, StatusCode code) {});
    callbackManager->reset();
    EXPECT_NO_THROW(callbackManager->runCallback());
}
