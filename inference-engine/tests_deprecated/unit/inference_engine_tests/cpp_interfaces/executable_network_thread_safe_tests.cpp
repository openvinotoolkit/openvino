// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "cpp_interfaces/base/ie_executable_network_base.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_thread_safe_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_infer_request_internal.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class ExecutableNetworkThreadSafeTests : public ::testing::Test {
protected:
    shared_ptr<MockExecutableNetworkThreadSafe> mockExeNetwork;
    shared_ptr<IExecutableNetwork> exeNetwork;
    shared_ptr<MockInferRequestInternal> mockInferRequestInternal;
    ResponseDesc dsc;
    StatusCode sts;

    virtual void TearDown() {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockInferRequestInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetwork.get()));
    }

    virtual void SetUp() {
        mockExeNetwork = make_shared<MockExecutableNetworkThreadSafe>();
        exeNetwork = details::shared_from_irelease(
                new ExecutableNetworkBase<MockExecutableNetworkThreadSafe>(mockExeNetwork));
        InputsDataMap networkInputs;
        OutputsDataMap networkOutputs;
        mockInferRequestInternal = make_shared<MockInferRequestInternal>(networkInputs, networkOutputs);
    }
};

TEST_F(ExecutableNetworkThreadSafeTests, createInferRequestCallsThreadSafeImplAndSetNetworkIO) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    auto threadSafeReq = dynamic_pointer_cast<InferRequestBase<AsyncInferRequestThreadSafeDefault>>(req);
    ASSERT_NE(threadSafeReq, nullptr);
}

TEST_F(ExecutableNetworkThreadSafeTests, returnErrorIfInferThrowsException) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mockExeNetwork.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
    EXPECT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
    EXPECT_CALL(*mockInferRequestInternal.get(), InferImpl()).WillOnce(Throw(std::runtime_error("")));
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
    EXPECT_NO_THROW(sts = req->Wait(IInferRequest::WaitMode::RESULT_READY, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << dsc.msg;
}
