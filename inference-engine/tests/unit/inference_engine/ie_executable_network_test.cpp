// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "cpp/ie_executable_network.hpp"
#include "ie_iexecutable_network.hpp"
#include "ie_plugin_cpp.hpp"

#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_ie_ivariable_state.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"

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


class ExecutableNetworkTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIExecutableNetworkInternal> mockIExeNet;
    InferenceEngine::ExecutableNetwork exeNetwork;

    struct TestPluginInternal : public MockIInferencePlugin {
        TestPluginInternal(const std::shared_ptr<MockIExecutableNetworkInternal>& mockIExeNet_) : mockIExeNet{mockIExeNet_} {}
        std::shared_ptr<IExecutableNetworkInternal> LoadNetwork(const CNNNetwork&, const std::map<std::string, std::string>&) override {
            return mockIExeNet;
        }
        QueryNetworkResult QueryNetwork(const CNNNetwork&, const std::map<std::string, std::string>&) const override {
            IE_THROW(NotImplemented);
        }
        std::shared_ptr<MockIExecutableNetworkInternal> mockIExeNet;
    };
    struct TestPlugin : public InferenceEngine::InferencePlugin {
        TestPlugin(std::shared_ptr<MockIExecutableNetworkInternal> mockIExeNet) :
            InferenceEngine::InferencePlugin{InferenceEngine::details::SOPointer<TestPluginInternal>{
                new TestPluginInternal{mockIExeNet}}} {}
    };

    virtual void TearDown() {
        mockIExeNet.reset();
        exeNetwork = {};
    }

    virtual void SetUp() {
        mockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
        exeNetwork = TestPlugin{mockIExeNet}.LoadNetwork({}, {});
    }
};

TEST_F(ExecutableNetworkTests, GetOutputsInfoThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet.get(), GetOutputsInfo())
            .Times(1)
            .WillOnce(Throw(InferenceEngine::GeneralError{""}));

    ASSERT_THROW(exeNetwork.GetOutputsInfo(), InferenceEngine::Exception);
}

TEST_F(ExecutableNetworkTests, GetOutputsInfo) {
    InferenceEngine::ConstOutputsDataMap data;
    EXPECT_CALL(*mockIExeNet.get(), GetOutputsInfo()).Times(1).WillRepeatedly(Return(InferenceEngine::ConstOutputsDataMap{}));

    ASSERT_NO_THROW(data = exeNetwork.GetOutputsInfo());
    ASSERT_EQ(data, InferenceEngine::ConstOutputsDataMap{});
}

TEST_F(ExecutableNetworkTests, GetInputsInfoThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet.get(), GetInputsInfo())
            .Times(1)
            .WillOnce(Throw(InferenceEngine::GeneralError{""}));

    ASSERT_THROW(exeNetwork.GetInputsInfo(), InferenceEngine::Exception);
}

TEST_F(ExecutableNetworkTests, GetInputsInfo) {
    EXPECT_CALL(*mockIExeNet.get(), GetInputsInfo()).Times(1).WillRepeatedly(Return(InferenceEngine::ConstInputsDataMap{}));

    InferenceEngine::ConstInputsDataMap info;
    ASSERT_NO_THROW(info = exeNetwork.GetInputsInfo());
    ASSERT_EQ(info, InferenceEngine::ConstInputsDataMap{});
}


TEST_F(ExecutableNetworkTests, resetThrowsIfResetToNullptr) {
    InferenceEngine::IExecutableNetwork::Ptr mockIExeNet_2{};
    ASSERT_THROW(exeNetwork.reset(mockIExeNet_2), InferenceEngine::Exception);
}

IE_SUPPRESS_DEPRECATED_START
TEST_F(ExecutableNetworkTests, QueryStateThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet.get(), QueryState())
            .Times(1)
            .WillOnce(Throw(InferenceEngine::GeneralError{""}));
    EXPECT_THROW(exeNetwork.QueryState(), InferenceEngine::Exception);
}

TEST_F(ExecutableNetworkTests, QueryState) {
    auto mockIMemState_p = std::make_shared<MockIVariableStateInternal>();
    EXPECT_CALL(*mockIExeNet.get(), QueryState())
            .Times(1)
            .WillOnce(Return(std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>>(1, mockIMemState_p)));
    std::vector<InferenceEngine::VariableState> MemState_v;
    EXPECT_NO_THROW(MemState_v = exeNetwork.QueryState());
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
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Return(mockIInferReq_p));
    InferRequest actualInferReq;
    ASSERT_NO_THROW(actualInferReq = exeNetwork.CreateInferRequest());
    ASSERT_EQ(mockIInferReq_p, static_cast<IInferRequest::Ptr &>(actualInferReq));
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestThrowsIfReturnNotOK) {
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Throw(InferenceEngine::GeneralError{""}));
    ASSERT_THROW(exeNetwork.CreateInferRequest(), InferenceEngine::Exception);
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestThrowsIfSetRequestToNullptr) {
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest())
            .WillOnce(Return(std::shared_ptr<MockIInferRequest>{}));
    ASSERT_THROW(exeNetwork.CreateInferRequest(), InferenceEngine::Exception);
}

// CreateInferRequestPtr
TEST_F(ExecutableNetworkWithIInferReqTests, CanCreateInferRequestPtr) {
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Return(mockIInferReq_p));
    InferRequest::Ptr actualInferReq;
    ASSERT_NO_THROW(actualInferReq = exeNetwork.CreateInferRequestPtr());
    ASSERT_EQ(mockIInferReq_p, static_cast<IInferRequest::Ptr &>(*actualInferReq.get()));
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestPtrThrowsIfReturnNotOK) {
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Throw(InferenceEngine::GeneralError{""}));
    ASSERT_THROW(exeNetwork.CreateInferRequestPtr(), InferenceEngine::Exception);
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestPtrThrowsIfSetRequestToNullptr) {
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Return(std::shared_ptr<MockIInferRequest>{}));
    ASSERT_THROW(exeNetwork.CreateInferRequestPtr(), InferenceEngine::Exception);
}

IE_SUPPRESS_DEPRECATED_START

class ExecutableNetworkBaseTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIExecutableNetworkInternal> mock_impl;
    std::shared_ptr<IExecutableNetwork> exeNetwork;
    ResponseDesc dsc;

    virtual void TearDown() {
    }

    virtual void SetUp() {
        mock_impl.reset(new MockIExecutableNetworkInternal());
        exeNetwork = std::make_shared<ExecutableNetworkBase>(mock_impl);
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


