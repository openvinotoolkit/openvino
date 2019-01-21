// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <mock_not_empty_icnn_network.hpp>
#include <mock_iexecutable_network.hpp>
#include <mock_iasync_infer_request.hpp>
#include <cpp_interfaces/impl/mock_executable_network_internal.hpp>

#include <cpp/ie_executable_network.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class ExecutableNetworkTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIExecutableNetwork> mock_exe_network;
    ExecutableNetwork exeNetworkWrapper;
    std::shared_ptr<MockIInferRequest> mock_async_request;
    ResponseDesc dsc;

    virtual void TearDown() {
    }

    virtual void SetUp() {
        mock_exe_network = make_shared<MockIExecutableNetwork>();
        mock_async_request = make_shared<MockIInferRequest>();
        exeNetworkWrapper = ExecutableNetwork(mock_exe_network);
    }
};

// CreateInferRequest
TEST_F(ExecutableNetworkTests, canForwardCreateInferRequest) {
    EXPECT_CALL(*mock_exe_network.get(), CreateInferRequest(_, _))
            .WillOnce(DoAll(SetArgReferee<0>(mock_async_request), Return(OK)));
    InferRequest actual_async_request;
    ASSERT_NO_THROW(actual_async_request = exeNetworkWrapper.CreateInferRequest());
    ASSERT_EQ(mock_async_request, static_cast<IInferRequest::Ptr &>(actual_async_request));
}

TEST_F(ExecutableNetworkTests, throwsIfCreateInferRequestReturnNotOK) {
    EXPECT_CALL(*mock_exe_network.get(), CreateInferRequest(_, _)).WillOnce(Return(GENERAL_ERROR));
    ASSERT_THROW(exeNetworkWrapper.CreateInferRequest(), InferenceEngineException);
}

// CreateInferRequestPtr
TEST_F(ExecutableNetworkTests, canForwardCreateInferRequestPtr) {
    EXPECT_CALL(*mock_exe_network.get(), CreateInferRequest(_, _))
            .WillOnce(DoAll(SetArgReferee<0>(mock_async_request), Return(OK)));
    InferRequest::Ptr actual_async_request;
    ASSERT_NO_THROW(actual_async_request = exeNetworkWrapper.CreateInferRequestPtr());
    ASSERT_EQ(mock_async_request, static_cast<IInferRequest::Ptr &>(*actual_async_request.get()));
}

TEST_F(ExecutableNetworkTests, throwsIfCreateInferRequestPtrReturnNotOK) {
    EXPECT_CALL(*mock_exe_network.get(), CreateInferRequest(_, _)).WillOnce(Return(GENERAL_ERROR));
    ASSERT_THROW(exeNetworkWrapper.CreateInferRequest(), InferenceEngineException);
}
