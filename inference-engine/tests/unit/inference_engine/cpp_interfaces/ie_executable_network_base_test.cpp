// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include <cpp_interfaces/base/ie_executable_network_base.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_thread_safe_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

IE_SUPPRESS_DEPRECATED_START

class ExecutableNetworkThreadSafeTests : public ::testing::Test {
protected:
    shared_ptr<MockExecutableNetworkThreadSafe> mockExeNetwork;
    shared_ptr<IExecutableNetwork> exeNetwork;
    shared_ptr<MockIInferRequestInternal> mockInferRequestInternal;
    ResponseDesc dsc;
    StatusCode sts;

    virtual void TearDown() {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockInferRequestInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetwork.get()));
    }

    virtual void SetUp() {
        mockExeNetwork = make_shared<MockExecutableNetworkThreadSafe>();
        exeNetwork = std::make_shared<ExecutableNetworkBase>(mockExeNetwork);
        InputsDataMap networkInputs;
        OutputsDataMap networkOutputs;
        mockInferRequestInternal = make_shared<MockIInferRequestInternal>(networkInputs, networkOutputs);
    }
};

TEST_F(ExecutableNetworkThreadSafeTests, createInferRequestCallsThreadSafeImplAndSetNetworkIO) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    auto threadSafeReq = dynamic_pointer_cast<InferRequestBase>(req);
    ASSERT_NE(threadSafeReq, nullptr);
}

TEST_F(ExecutableNetworkThreadSafeTests, returnErrorIfInferThrowsException) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    EXPECT_CALL(*mockInferRequestInternal.get(), checkBlobs()).WillOnce(Throw(std::runtime_error("")));
    EXPECT_NO_THROW(sts = req->Infer(&dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << dsc.msg;
}

TEST_F(ExecutableNetworkThreadSafeTests, returnErrorIfStartAsyncThrowsException) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    EXPECT_CALL(*mockInferRequestInternal.get(), InferImpl()).WillOnce(Throw(std::runtime_error("")));
    EXPECT_NO_THROW(sts = req->StartAsync(&dsc));
    ASSERT_TRUE(StatusCode::OK == sts) << dsc.msg;
    EXPECT_NO_THROW(sts = req->Wait(InferRequest::WaitMode::RESULT_READY, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << dsc.msg;
}
