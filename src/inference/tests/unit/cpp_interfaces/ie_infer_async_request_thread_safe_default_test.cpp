// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-spec-builders.h>
#include <gtest/gtest.h>

#include <cpp/ie_infer_async_request_base.hpp>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include <deque>
#include <inference_engine.hpp>

#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_async_infer_request_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/mock_task_executor.hpp"

using namespace ::testing;
using namespace std;
// using namespace InferenceEngine;
// using namespace InferenceEngine::details;
using namespace ov::threading;

struct DeferedExecutor : public ITaskExecutor {
    using Ptr = std::shared_ptr<DeferedExecutor>;
    DeferedExecutor() = default;

    void executeOne() {
        tasks.front()();
        tasks.pop_front();
    }

    void executeAll() {
        while (!tasks.empty()) {
            executeOne();
        }
    }

    ~DeferedExecutor() override {
        executeAll();
    };

    void run(Task task) override {
        tasks.push_back(task);
    }

    std::deque<Task> tasks;
};

class InferRequestThreadSafeDefaultTests : public ::testing::Test {
protected:
    shared_ptr<AsyncInferRequestThreadSafeDefault> testRequest;
    ResponseDesc dsc;

    shared_ptr<MockIInferRequestInternal> mockInferRequestInternal;
    MockTaskExecutor::Ptr mockTaskExecutor;

    void TearDown() override {}

    void SetUp() override {
        InputsDataMap inputsInfo;
        OutputsDataMap outputsInfo;
        mockTaskExecutor = make_shared<MockTaskExecutor>();
        mockInferRequestInternal = make_shared<MockIInferRequestInternal>(inputsInfo, outputsInfo);
        testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal,
                                                                      mockTaskExecutor,
                                                                      mockTaskExecutor);
    }
};

// StartAsync
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnStartAsync) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_THROW(testRequest->StartAsync(), RequestBusy);
    taskExecutor->executeAll();
}

TEST_F(InferRequestThreadSafeDefaultTests, canResetBusyStatusIfStartAsyncFails) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, checkBlobs()).Times(2).WillOnce(Throw(GeneralError{""})).WillOnce(Return());

    ASSERT_THROW(testRequest->StartAsync(), GeneralError);
    ASSERT_NO_THROW(testRequest->StartAsync());
    taskExecutor->executeAll();
}

// Wait
TEST_F(InferRequestThreadSafeDefaultTests, returnInferNotStartedOnWait) {
    int64_t ms = 0;
    StatusCode actual = testRequest->Wait(ms);
    ASSERT_EQ(INFER_NOT_STARTED, actual);
}

// Infer
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnInfer) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_THROW(testRequest->StartAsync(), RequestBusy);
    taskExecutor->executeAll();
}

TEST_F(InferRequestThreadSafeDefaultTests, canResetBusyStatusIfInferFails) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(2).WillOnce(Throw(GeneralError{""})).WillOnce(Return());
    ASSERT_THROW(testRequest->Infer(), GeneralError);

    ASSERT_NO_THROW(testRequest->Infer());
    taskExecutor->executeAll();
}

// GetPerformanceCounts
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnGetPerformanceCounts) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_THROW(testRequest->GetPerformanceCounts(), RequestBusy);
    taskExecutor->executeAll();
}

// GetBlob
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnGetBlob) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_THROW(testRequest->GetBlob({}), RequestBusy);
    taskExecutor->executeAll();
}

// SetBlob
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnSetBlob) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_THROW(testRequest->SetBlob({}, {}), RequestBusy);
    taskExecutor->executeAll();
}

// SetCompletionCallback
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnSetCompletionCallback) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_THROW(testRequest->SetCallback({}), RequestBusy);
    taskExecutor->executeAll();
}

TEST_F(InferRequestThreadSafeDefaultTests, callbackTakesOKIfAsyncRequestWasOK) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    std::exception_ptr exceptionPtr;
    testRequest->SetCallback([&](std::exception_ptr exceptionPtr_) {
        exceptionPtr = exceptionPtr_;
    });
    EXPECT_CALL(*mockInferRequestInternal.get(), InferImpl()).Times(1);
    testRequest->StartAsync();
    taskExecutor->executeAll();
    testRequest->Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(nullptr, exceptionPtr);
}

TEST_F(InferRequestThreadSafeDefaultTests, callbackIsCalledIfAsyncRequestFailed) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    std::exception_ptr exceptionPtr;
    testRequest->SetCallback([&](std::exception_ptr exceptionPtr_) {
        exceptionPtr = exceptionPtr_;
    });
    EXPECT_CALL(*mockInferRequestInternal.get(), InferImpl()).WillOnce(Throw(std::exception()));
    testRequest->StartAsync();
    taskExecutor->executeAll();
    EXPECT_THROW(testRequest->Wait(InferRequest::WaitMode::RESULT_READY), std::exception);
    ASSERT_NE(nullptr, exceptionPtr);
}

TEST_F(InferRequestThreadSafeDefaultTests, canCatchExceptionIfAsyncRequestFailedAndNoCallback) {
    auto taskExecutor = std::make_shared<CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{});
    testRequest = make_shared<AsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal.get(), InferImpl()).WillOnce(Throw(std::exception()));
    testRequest->StartAsync();
    EXPECT_THROW(testRequest->Wait(InferRequest::WaitMode::RESULT_READY), std::exception);
}
