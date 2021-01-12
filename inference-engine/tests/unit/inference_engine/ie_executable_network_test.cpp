// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "cpp/ie_executable_network.hpp"

#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_ie_ivariable_state.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"

using testing::_;
using testing::Throw;
using testing::Ref;
using testing::Return;
using testing::SetArgReferee;

// TODO: add tests for the next methods:
//  1. void Export(const std::string& modelFileName)
//  2. void Export(std::ostream& networkModel)
//  4. CNNNetwork GetExecGraphInfo()
//  5. void SetConfig(const std::map<std::string, Parameter>& config)
//  6. Parameter GetConfig(const std::string& name) const
//  7. Parameter GetMetric(const std::string& name) const
//  8. RemoteContext::Ptr GetContext()


TEST(ExecutableNetworkConstructorTests, ThrowsIfConstructFromNullptr) {
    // TODO issue: 26390; ExecutableNetwork's constructor shouldn't be available
    EXPECT_NO_THROW(InferenceEngine::ExecutableNetwork exeNet{});

    EXPECT_THROW(InferenceEngine::ExecutableNetwork exeNet{nullptr}, InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkConstructorTests, CanConstruct) {
    std::shared_ptr<MockIExecutableNetwork> mockIExeNet_p = std::make_shared<MockIExecutableNetwork>();
    InferenceEngine::ExecutableNetwork exeNet{mockIExeNet_p};
}

TEST(ExecutableNetworkDestructorTests, Destruct) {
    std::shared_ptr<MockIExecutableNetwork> mockIExeNet_p = std::make_shared<MockIExecutableNetwork>();
    InferenceEngine::ExecutableNetwork exeNet{mockIExeNet_p};
    exeNet.~ExecutableNetwork();
    // Call of destructor should decrease counter of shared_ptr
    ASSERT_EQ(mockIExeNet_p.use_count(), 1);
}

class ExecutableNetworkTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIExecutableNetwork> mockIExeNet_p;
    std::unique_ptr<InferenceEngine::ExecutableNetwork> exeNetwork;

    virtual void TearDown() {
        mockIExeNet_p.reset();
        exeNetwork.reset();
    }

    virtual void SetUp() {
        mockIExeNet_p = std::make_shared<MockIExecutableNetwork>();
        ASSERT_EQ(exeNetwork, nullptr);
        exeNetwork = std::unique_ptr<InferenceEngine::ExecutableNetwork>(
                new InferenceEngine::ExecutableNetwork(mockIExeNet_p));
        ASSERT_NE(exeNetwork, nullptr);
    }
};

TEST_F(ExecutableNetworkTests, GetOutputsInfoThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet_p.get(), GetOutputsInfo(_, _))
            .Times(1)
            .WillOnce(Return(InferenceEngine::GENERAL_ERROR));

    ASSERT_THROW(exeNetwork->GetOutputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, GetOutputsInfo) {
    EXPECT_CALL(*mockIExeNet_p.get(), GetOutputsInfo(_, _))
            .Times(1)
            .WillOnce(Return(InferenceEngine::OK));

    InferenceEngine::ConstOutputsDataMap data;
    ASSERT_NO_THROW(data = exeNetwork->GetOutputsInfo());
    ASSERT_EQ(data, InferenceEngine::ConstOutputsDataMap{});
}

TEST_F(ExecutableNetworkTests, GetInputsInfoThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet_p.get(), GetInputsInfo(_, _))
            .Times(1)
            .WillOnce(Return(InferenceEngine::GENERAL_ERROR));

    ASSERT_THROW(exeNetwork->GetInputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, GetInputsInfo) {
    EXPECT_CALL(*mockIExeNet_p.get(), GetInputsInfo(_, _))
            .Times(1)
            .WillOnce(Return(InferenceEngine::OK));

    InferenceEngine::ConstInputsDataMap info;
    ASSERT_NO_THROW(info = exeNetwork->GetInputsInfo());
    ASSERT_EQ(info, InferenceEngine::ConstInputsDataMap{});
}


TEST_F(ExecutableNetworkTests, resetThrowsIfResetToNullptr) {
    InferenceEngine::IExecutableNetwork::Ptr mockIExeNet_p_2{};
    ASSERT_THROW(exeNetwork->reset(mockIExeNet_p_2), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, reset) {
    InferenceEngine::IExecutableNetwork::Ptr mockIExeNet_p_2 = std::make_shared<MockIExecutableNetwork>();

    exeNetwork->reset(mockIExeNet_p_2);

    InferenceEngine::IExecutableNetwork::Ptr exeNet_p = *exeNetwork;   // use of IExecutableNetwork::Ptr&
    EXPECT_NE(exeNet_p, mockIExeNet_p);
    EXPECT_EQ(exeNet_p, mockIExeNet_p_2);
}

TEST_F(ExecutableNetworkTests, OperatorAmpersand) {
    InferenceEngine::IExecutableNetwork::Ptr exeNet_p = *exeNetwork;   // use of IExecutableNetwork::Ptr&
    ASSERT_EQ(exeNet_p, mockIExeNet_p);
}

IE_SUPPRESS_DEPRECATED_START
TEST_F(ExecutableNetworkTests, QueryStateThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet_p.get(), QueryState(_, _, _))
            .Times(1)
            .WillOnce(Return(InferenceEngine::GENERAL_ERROR));
    EXPECT_THROW(exeNetwork->QueryState(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, QueryStateIfReturnOutOfBounds) {
    EXPECT_CALL(*mockIExeNet_p.get(), QueryState(_, _, _))
            .Times(1)
            .WillOnce(Return(InferenceEngine::OUT_OF_BOUNDS));
    std::vector<InferenceEngine::VariableState> MemState_;
    EXPECT_NO_THROW(MemState_ = exeNetwork->QueryState());
    EXPECT_EQ(MemState_.size(), 0);
}

TEST_F(ExecutableNetworkTests, QueryState) {
    std::shared_ptr<MockIVariableState> mockIMemState_p = std::make_shared<MockIVariableState>();
    EXPECT_CALL(*mockIExeNet_p.get(), QueryState(_, _, _))
            .Times(2)
            .WillOnce(DoAll(SetArgReferee<0>(mockIMemState_p), Return(InferenceEngine::OK)))
            .WillOnce(Return(InferenceEngine::OUT_OF_BOUNDS));
    std::vector<InferenceEngine::VariableState> MemState_v;
    EXPECT_NO_THROW(MemState_v = exeNetwork->QueryState());
    EXPECT_EQ(MemState_v.size(), 1);
}
IE_SUPPRESS_DEPRECATED_END

class ExecutableNetworkWithIInferReqTests : public ExecutableNetworkTests {
protected:
    std::shared_ptr<MockIInferRequest> mockIInferReq_p;

    virtual void TearDown() {
        ExecutableNetworkTests::TearDown();
        mockIInferReq_p.reset();
    }

    virtual void SetUp() {
        ExecutableNetworkTests::SetUp();
        mockIInferReq_p = std::make_shared<MockIInferRequest>();
    }
};

TEST_F(ExecutableNetworkWithIInferReqTests, CanCreateInferRequest) {
    EXPECT_CALL(*mockIExeNet_p.get(), CreateInferRequest(_, _))
            .WillOnce(DoAll(SetArgReferee<0>(mockIInferReq_p), Return(InferenceEngine::OK)));
    InferRequest actualInferReq;
    ASSERT_NO_THROW(actualInferReq = exeNetwork->CreateInferRequest());
    ASSERT_EQ(mockIInferReq_p, static_cast<IInferRequest::Ptr &>(actualInferReq));
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestThrowsIfReturnNotOK) {
    EXPECT_CALL(*mockIExeNet_p.get(), CreateInferRequest(_, _)).WillOnce(Return(InferenceEngine::GENERAL_ERROR));
    ASSERT_THROW(exeNetwork->CreateInferRequest(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestThrowsIfSetRequestToNullptr) {
    EXPECT_CALL(*mockIExeNet_p.get(), CreateInferRequest(_, _))
            .WillOnce(DoAll(SetArgReferee<0>(nullptr), Return(InferenceEngine::OK)));
    ASSERT_THROW(exeNetwork->CreateInferRequest(), InferenceEngine::details::InferenceEngineException);
}

// CreateInferRequestPtr
TEST_F(ExecutableNetworkWithIInferReqTests, CanCreateInferRequestPtr) {
    EXPECT_CALL(*mockIExeNet_p.get(), CreateInferRequest(_, _))
            .WillOnce(DoAll(SetArgReferee<0>(mockIInferReq_p), Return(InferenceEngine::OK)));
    InferRequest::Ptr actualInferReq;
    ASSERT_NO_THROW(actualInferReq = exeNetwork->CreateInferRequestPtr());
    ASSERT_EQ(mockIInferReq_p, static_cast<IInferRequest::Ptr &>(*actualInferReq.get()));
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestPtrThrowsIfReturnNotOK) {
    EXPECT_CALL(*mockIExeNet_p.get(), CreateInferRequest(_, _)).WillOnce(Return(InferenceEngine::GENERAL_ERROR));
    ASSERT_THROW(exeNetwork->CreateInferRequestPtr(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestPtrThrowsIfSetRequestToNullptr) {
    EXPECT_CALL(*mockIExeNet_p.get(), CreateInferRequest(_, _))
            .WillOnce(DoAll(SetArgReferee<0>(nullptr), Return(InferenceEngine::OK)));
    ASSERT_THROW(exeNetwork->CreateInferRequestPtr(), InferenceEngine::details::InferenceEngineException);
}

class ExecutableNetworkBaseTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIExecutableNetworkInternal> mock_impl;
    std::shared_ptr<IExecutableNetwork> exeNetwork;
    ResponseDesc dsc;

    virtual void TearDown() {
    }

    virtual void SetUp() {
        mock_impl.reset(new MockIExecutableNetworkInternal());
        exeNetwork = shared_from_irelease(new ExecutableNetworkBase<MockIExecutableNetworkInternal>(mock_impl));
    }
};

// CreateInferRequest
TEST_F(ExecutableNetworkBaseTests, canForwardCreateInferRequest) {
    IInferRequest::Ptr req;
    EXPECT_CALL(*mock_impl.get(), CreateInferRequest()).Times(1).WillRepeatedly(Return(req));
    ASSERT_EQ(OK, exeNetwork->CreateInferRequest(req, &dsc));
}

TEST_F(ExecutableNetworkBaseTests, canReportErrorInCreateInferRequest) {
    EXPECT_CALL(*mock_impl.get(), CreateInferRequest()).WillOnce(Throw(std::runtime_error("compare")));
    IInferRequest::Ptr req;
    ASSERT_NE(OK, exeNetwork->CreateInferRequest(req, &dsc));
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(ExecutableNetworkBaseTests, canCatchUnknownErrorInCreateInferRequest) {
    EXPECT_CALL(*mock_impl.get(), CreateInferRequest()).WillOnce(Throw(5));
    IInferRequest::Ptr req;
    ASSERT_EQ(UNEXPECTED, exeNetwork->CreateInferRequest(req, nullptr));
}

// Export
TEST_F(ExecutableNetworkBaseTests, canForwardExport) {
    const std::string modelFileName;
    EXPECT_CALL(*mock_impl.get(), Export(Ref(modelFileName))).Times(1);
    ASSERT_EQ(OK, exeNetwork->Export(modelFileName, &dsc));
}

TEST_F(ExecutableNetworkBaseTests, canReportErrorInExport) {
    EXPECT_CALL(*mock_impl.get(), Export(_)).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(exeNetwork->Export({}, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(ExecutableNetworkBaseTests, canCatchUnknownErrorInExport) {
    EXPECT_CALL(*mock_impl.get(), Export(_)).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, exeNetwork->Export({}, nullptr));
}


