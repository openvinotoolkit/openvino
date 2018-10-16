// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <mock_mkldnn_extension.hpp>
#include <cpp_interfaces/impl/mock_executable_thread_safe_async_only.hpp>
#include <cpp_interfaces/impl/mock_async_infer_request_internal.hpp>

#include <ie_version.hpp>
#include <cpp_interfaces/base/ie_executable_network_base.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class ExecutableNetworkThreadSafeAsyncOnlyTests : public ::testing::Test {
protected:
    shared_ptr<MockExecutableNetworkThreadSafeAsyncOnly> mockExeNetwork;
    shared_ptr<MockAsyncInferRequestInternal> mockAsyncInferRequestInternal;
    shared_ptr<IExecutableNetwork> exeNetwork;
    ResponseDesc dsc;
    StatusCode sts;

    virtual void TearDown() {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockAsyncInferRequestInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetwork.get()));
    }

    virtual void SetUp() {
        mockExeNetwork = make_shared<MockExecutableNetworkThreadSafeAsyncOnly>();
        exeNetwork = details::shared_from_irelease(
                new ExecutableNetworkBase<MockExecutableNetworkThreadSafeAsyncOnly>(mockExeNetwork));
        InputsDataMap networkInputs;
        OutputsDataMap networkOutputs;
        mockAsyncInferRequestInternal = make_shared<MockAsyncInferRequestInternal>(networkInputs, networkOutputs);
    }
};

TEST_F(ExecutableNetworkThreadSafeAsyncOnlyTests, createAsyncInferRequestCallsThreadSafeImplAndSetNetworkIO) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateAsyncInferRequestImpl(_, _)).WillOnce(
            Return(mockAsyncInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    auto threadSafeReq = dynamic_pointer_cast<InferRequestBase<AsyncInferRequestInternal>>(req);
    ASSERT_NE(threadSafeReq, nullptr);
}

TEST_F(ExecutableNetworkThreadSafeAsyncOnlyTests, returnErrorIfInferThrowsException) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateAsyncInferRequestImpl(_, _)).WillOnce(
            Return(mockAsyncInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    EXPECT_CALL(*mockAsyncInferRequestInternal.get(), InferImpl()).WillOnce(Throw(std::runtime_error("")));
    EXPECT_NO_THROW(sts = req->Infer(&dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << dsc.msg;
}

TEST_F(ExecutableNetworkThreadSafeAsyncOnlyTests, returnErrorIfStartAsyncThrowsException) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateAsyncInferRequestImpl(_, _)).WillOnce(
            Return(mockAsyncInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    EXPECT_CALL(*mockAsyncInferRequestInternal.get(), StartAsyncImpl()).WillOnce(Throw(std::runtime_error("")));
    EXPECT_NO_THROW(sts = req->StartAsync(&dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << dsc.msg;
}

TEST_F(ExecutableNetworkThreadSafeAsyncOnlyTests, canForwardStartAsyncAndInfer) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateAsyncInferRequestImpl(_, _)).WillOnce(
            Return(mockAsyncInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    EXPECT_CALL(*mockAsyncInferRequestInternal.get(), StartAsyncImpl()).Times(1);
    EXPECT_CALL(*mockAsyncInferRequestInternal.get(), InferImpl()).Times(1);

    EXPECT_NO_THROW(req->StartAsync(&dsc)) << dsc.msg;
    EXPECT_NO_THROW(req->Infer(&dsc)) << dsc.msg;
}

TEST_F(ExecutableNetworkThreadSafeAsyncOnlyTests, canForwardInferAndStartAsync) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateAsyncInferRequestImpl(_, _)).WillOnce(
            Return(mockAsyncInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    EXPECT_CALL(*mockAsyncInferRequestInternal.get(), StartAsyncImpl()).Times(1);
    EXPECT_CALL(*mockAsyncInferRequestInternal.get(), InferImpl()).Times(1);
    EXPECT_NO_THROW(req->Infer(&dsc)) << dsc.msg;
    EXPECT_NO_THROW(req->StartAsync(&dsc)) << dsc.msg;
}
