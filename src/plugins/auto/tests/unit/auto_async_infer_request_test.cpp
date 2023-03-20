// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "include/mock_auto_device_plugin.hpp"
#include "include/auto_infer_request_test_base.hpp"

void AsyncInferenceTest::SetUp() {
        ON_CALL(*core.get(), isNewAPI()).WillByDefault(Return(true));
        // replace core with mock Icore
        plugin->SetCore(core);
        makeAsyncRequest();
        exeNetwork = plugin->LoadNetwork(cnnNet, {});
        auto_request = exeNetwork->CreateInferRequest();
}

void AsyncInferenceTest::makeAsyncRequest() {
    taskExecutor = std::make_shared<DeferedExecutor>();
    // set up mock infer request
    for (size_t i = 0; i < target_request_num; i++) {
        auto inferReq = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        auto asyncRequest = std::make_shared<AsyncInferRequestThreadSafeDefault>(inferReq, taskExecutor, taskExecutor);
        inferReqInternal.push_back(inferReq);
        asyncInferRequest.push_back(asyncRequest);
    }
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Return(asyncInferRequest[0]))
                                                            .WillOnce(Return(asyncInferRequest[1]))
                                                            .WillOnce(Return(asyncInferRequest[2]))
                                                            .WillOnce(Return(asyncInferRequest[3]))
                                                            .WillRepeatedly(Return(asyncInferRequest[0]));
}

void AsyncInferenceTest::TearDown() {
}

TEST_F(AsyncInferenceTest, returnRequestBusyOnStartAsync) {
    for (size_t i = 0; i < target_request_num ; i++)
        ON_CALL(*inferReqInternal[i], InferImpl()).WillByDefault(Return());
    ASSERT_NO_THROW(auto_request->StartAsync());
    ASSERT_THROW(auto_request->StartAsync(), RequestBusy);
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
}

TEST_F(AsyncInferenceTest, canInferIfStartAsyncSuccess) {
    auto_request->SetCallback([&](std::exception_ptr exceptionPtr_) {
        auto exceptionPtr = exceptionPtr_;
        ASSERT_EQ(exceptionPtr, nullptr);
    });
    auto_request->StartAsync();
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
    ASSERT_NO_THROW(auto_request->Wait(InferRequest::WaitMode::RESULT_READY));
}

TEST_F(AsyncInferenceTest, canRethrowIfStartAsyncFails) {
    for (size_t i = 0; i < target_request_num ; i++)
        ON_CALL(*inferReqInternal[i], InferImpl()).WillByDefault(Throw(std::exception()));
    auto_request->SetCallback([&](std::exception_ptr exceptionPtr_) {
        auto exceptionPtr = exceptionPtr_;
        ASSERT_NE(exceptionPtr, nullptr);
    });
    auto_request->StartAsync();
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
    EXPECT_THROW(auto_request->Wait(InferRequest::WaitMode::RESULT_READY), std::exception);
}

TEST_F(AsyncInferenceTest, canStart2AsyncInferRequests) {
    auto another_auto_request = exeNetwork->CreateInferRequest();
    auto_request->SetCallback([&](std::exception_ptr exceptionPtr_) {
        auto exceptionPtr = exceptionPtr_;
        ASSERT_EQ(exceptionPtr, nullptr);
    });
    another_auto_request->SetCallback([&](std::exception_ptr exceptionPtr_) {
        auto exceptionPtr = exceptionPtr_;
        ASSERT_EQ(exceptionPtr, nullptr);
    });
    auto_request->StartAsync();
    another_auto_request->StartAsync();
    std::dynamic_pointer_cast<DeferedExecutor>(taskExecutor)->executeAll();
    ASSERT_NO_THROW(auto_request->Wait(InferRequest::WaitMode::RESULT_READY));
    ASSERT_NO_THROW(another_auto_request->Wait(InferRequest::WaitMode::RESULT_READY));
}