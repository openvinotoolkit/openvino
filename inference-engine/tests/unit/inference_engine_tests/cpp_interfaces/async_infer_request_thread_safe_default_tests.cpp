// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <inference_engine.hpp>
#include <cpp_interfaces/impl/mock_infer_request_internal.hpp>
#include <cpp_interfaces/impl/mock_async_infer_request_default.hpp>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include <cpp_interfaces/mock_task_executor.hpp>
#include <cpp_interfaces/base/ie_infer_async_request_base.hpp>
#include <deque>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class TestAsyncInferRequestThreadSafeDefault : public AsyncInferRequestThreadSafeDefault {
public:
    TestAsyncInferRequestThreadSafeDefault(const InferRequestInternal::Ptr &request,
                                           const ITaskExecutor::Ptr &taskExecutor,
                                           const ITaskExecutor::Ptr &callbackExecutor)
            : AsyncInferRequestThreadSafeDefault(request, taskExecutor, callbackExecutor) {}

    void setRequestBusy() {
        AsyncInferRequestThreadSafeDefault::setIsRequestBusy(true);
    }
};

struct DeferedExecutor : public ITaskExecutor {
    using Ptr = std::shared_ptr<DeferedExecutor>;
    DeferedExecutor() = default;

    void executeOne() {
        tasks.front()();
        tasks.pop_front();
    }

    void executeAll() {
        while(!tasks.empty()) {
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
    shared_ptr<TestAsyncInferRequestThreadSafeDefault> testRequest;
    ResponseDesc dsc;

    shared_ptr<MockInferRequestInternal> mockInferRequestInternal;
    MockTaskExecutor::Ptr mockTaskExecutor;


    virtual void TearDown() {
    }

    virtual void SetUp() {
        InputsDataMap inputsInfo;
        OutputsDataMap outputsInfo;
        mockTaskExecutor = make_shared<MockTaskExecutor>();
        mockInferRequestInternal = make_shared<MockInferRequestInternal>(inputsInfo, outputsInfo);
        testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, mockTaskExecutor, mockTaskExecutor);
    }

    bool _doesThrowExceptionWithMessage(std::function<void()> func, string refError) {
        std::string whatMessage;
        try {
            func();
        } catch (const InferenceEngineException &iee) {
            whatMessage = iee.what();
        }
        return whatMessage.find(refError) != std::string::npos;
    }
};

// StartAsync
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnStartAsync) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->StartAsync(); }, REQUEST_BUSY_str));
    taskExecutor->executeAll();
}

TEST_F(InferRequestThreadSafeDefaultTests, canResetBusyStatusIfStartAsyncFails) {
    MockAsyncInferRequestDefault mockAsync(mockInferRequestInternal, mockTaskExecutor, mockTaskExecutor);
    EXPECT_CALL(mockAsync, StartAsync_ThreadUnsafe()).Times(2)
            .WillOnce(Throw(InferenceEngineException(__FILE__, __LINE__) << "compare"))
            .WillOnce(Return());

    ASSERT_TRUE(_doesThrowExceptionWithMessage([&]() { mockAsync.StartAsync(); }, "compare"));
    ASSERT_NO_THROW(mockAsync.StartAsync());
}

// GetUserData
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnGetUserData) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->GetUserData(nullptr); }, REQUEST_BUSY_str));
    taskExecutor->executeAll();
}

// SetUserData
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnSetUserData) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->SetUserData(nullptr); }, REQUEST_BUSY_str));
    taskExecutor->executeAll();
}

// Wait
TEST_F(InferRequestThreadSafeDefaultTests, returnInferNotStartedOnWait) {
    testRequest->setRequestBusy();
    int64_t ms = 0;
    StatusCode actual = testRequest->Wait(ms);
    ASSERT_EQ(INFER_NOT_STARTED, actual);
}

// Infer
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnInfer) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->Infer(); }, REQUEST_BUSY_str));
    taskExecutor->executeAll();
}

TEST_F(InferRequestThreadSafeDefaultTests, canResetBusyStatusIfInferFails) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(2)
            .WillOnce(Throw(InferenceEngineException(__FILE__, __LINE__) << "compare"))
            .WillOnce(Return());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->Infer(); }, "compare"));
    ASSERT_NO_THROW(testRequest->Infer());
}

// GetPerformanceCounts
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnGetPerformanceCounts) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
        testRequest->GetPerformanceCounts(info);
    }, REQUEST_BUSY_str));
    taskExecutor->executeAll();
}

// GetBlob
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnGetBlob) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() {
        Blob::Ptr data;
        testRequest->GetBlob(nullptr, data);
    }, REQUEST_BUSY_str));
    taskExecutor->executeAll();
}

// SetBlob
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnSetBlob) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->SetBlob(nullptr, nullptr); }, REQUEST_BUSY_str));
    taskExecutor->executeAll();
}

// SetCompletionCallback
TEST_F(InferRequestThreadSafeDefaultTests, returnRequestBusyOnSetCompletionCallback) {
    auto taskExecutor = std::make_shared<DeferedExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    EXPECT_CALL(*mockInferRequestInternal, InferImpl()).Times(1).WillOnce(Return());
    ASSERT_NO_THROW(testRequest->StartAsync());
    ASSERT_TRUE(_doesThrowExceptionWithMessage([this]() { testRequest->SetCompletionCallback(nullptr); },
                                               REQUEST_BUSY_str));
    taskExecutor->executeAll();
}

TEST_F(InferRequestThreadSafeDefaultTests, callbackTakesOKIfAsyncRequestWasOK) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);

    IInferRequest::Ptr asyncRequest;
    asyncRequest.reset(new InferRequestBase<TestAsyncInferRequestThreadSafeDefault>(
            testRequest), [](IInferRequest *p) { p->Release(); });
    testRequest->SetPointerToPublicInterface(asyncRequest);

    testRequest->SetCompletionCallback([](InferenceEngine::IInferRequest::Ptr request, StatusCode status) {
        ASSERT_EQ((int) StatusCode::OK, status);
    });
    EXPECT_CALL(*mockInferRequestInternal.get(), InferImpl()).Times(1);

    testRequest->StartAsync();
    testRequest->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

TEST_F(InferRequestThreadSafeDefaultTests, callbackIsCalledIfAsyncRequestFailed) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    IInferRequest::Ptr asyncRequest;
    asyncRequest.reset(new InferRequestBase<TestAsyncInferRequestThreadSafeDefault>(
            testRequest), [](IInferRequest *p) { p->Release(); });
    testRequest->SetPointerToPublicInterface(asyncRequest);

    bool wasCalled = false;
    InferRequest cppRequest(asyncRequest);
    std::function<void(InferRequest, StatusCode)> callback =
            [&](InferRequest request, StatusCode status) {
                wasCalled = true;
                ASSERT_EQ(StatusCode::GENERAL_ERROR, status);
            };
    cppRequest.SetCompletionCallback(callback);
    EXPECT_CALL(*mockInferRequestInternal.get(), InferImpl()).WillOnce(Throw(std::exception()));

    testRequest->StartAsync();
    EXPECT_THROW(testRequest->Wait(IInferRequest::WaitMode::RESULT_READY), std::exception);
    ASSERT_TRUE(wasCalled);
}

TEST_F(InferRequestThreadSafeDefaultTests, canCatchExceptionIfAsyncRequestFailedAndNoCallback) {
    auto taskExecutor = std::make_shared<TaskExecutor>();
    testRequest = make_shared<TestAsyncInferRequestThreadSafeDefault>(mockInferRequestInternal, taskExecutor, taskExecutor);
    IInferRequest::Ptr asyncRequest;
    asyncRequest.reset(new InferRequestBase<TestAsyncInferRequestThreadSafeDefault>(
            testRequest), [](IInferRequest *p) { p->Release(); });
    testRequest->SetPointerToPublicInterface(asyncRequest);

    EXPECT_CALL(*mockInferRequestInternal.get(), InferImpl()).WillOnce(Throw(std::exception()));

    testRequest->StartAsync();
    EXPECT_THROW(testRequest->Wait(IInferRequest::WaitMode::RESULT_READY), std::exception);
}
