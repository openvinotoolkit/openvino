// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "cpp/ie_executable_network.hpp"
#include "cpp/ie_executable_network_base.hpp"
#include "cpp/ie_plugin.hpp"

#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_ie_ivariable_state.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"

using testing::_;
using testing::MatcherCast;
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
    InferenceEngine::SoExecutableNetworkInternal    exeNetwork;
    MockIInferencePlugin*                           mockIPlugin;
    InferencePlugin                                 plugin;

    virtual void TearDown() {
        mockIExeNet.reset();
        exeNetwork = {};
        plugin = {};
    }

    virtual void SetUp() {
        mockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
        auto mockIPluginPtr = std::make_shared<MockIInferencePlugin>();
        ON_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(mockIExeNet));
        plugin = InferenceEngine::InferencePlugin{{}, mockIPluginPtr};
        exeNetwork = plugin.LoadNetwork(CNNNetwork{}, {});
    }
};

TEST_F(ExecutableNetworkTests, GetOutputsInfoThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet.get(), GetOutputsInfo())
            .Times(1)
            .WillOnce(Throw(InferenceEngine::GeneralError{""}));

    ASSERT_THROW(exeNetwork->GetOutputsInfo(), InferenceEngine::Exception);
}

TEST_F(ExecutableNetworkTests, GetOutputsInfo) {
    InferenceEngine::ConstOutputsDataMap data;
    EXPECT_CALL(*mockIExeNet.get(), GetOutputsInfo()).Times(1).WillRepeatedly(Return(InferenceEngine::ConstOutputsDataMap{}));
    ASSERT_NO_THROW(data = exeNetwork->GetOutputsInfo());
    ASSERT_EQ(data, InferenceEngine::ConstOutputsDataMap{});
}

TEST_F(ExecutableNetworkTests, GetInputsInfoThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet.get(), GetInputsInfo())
            .Times(1)
            .WillOnce(Throw(InferenceEngine::GeneralError{""}));

    ASSERT_THROW(exeNetwork->GetInputsInfo(), InferenceEngine::Exception);
}

TEST_F(ExecutableNetworkTests, GetInputsInfo) {
    EXPECT_CALL(*mockIExeNet.get(), GetInputsInfo()).Times(1).WillRepeatedly(Return(InferenceEngine::ConstInputsDataMap{}));

    InferenceEngine::ConstInputsDataMap info;
    ASSERT_NO_THROW(info = exeNetwork->GetInputsInfo());
    ASSERT_EQ(info, InferenceEngine::ConstInputsDataMap{});
}

IE_SUPPRESS_DEPRECATED_START

TEST_F(ExecutableNetworkTests, QueryStateThrowsIfReturnErr) {
    EXPECT_CALL(*mockIExeNet.get(), QueryState())
            .Times(1)
            .WillOnce(Throw(InferenceEngine::GeneralError{""}));
    EXPECT_THROW(exeNetwork->QueryState(), InferenceEngine::Exception);
}

TEST_F(ExecutableNetworkTests, QueryState) {
    auto mockIMemState_p = std::make_shared<MockIVariableStateInternal>();
    EXPECT_CALL(*mockIExeNet.get(), QueryState())
            .Times(1)
            .WillOnce(Return(std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>>(1, mockIMemState_p)));
    std::vector<InferenceEngine::IVariableStateInternal::Ptr> MemState_v;
    EXPECT_NO_THROW(MemState_v = exeNetwork->QueryState());
    EXPECT_EQ(MemState_v.size(), 1);
}

IE_SUPPRESS_DEPRECATED_END

class ExecutableNetworkWithIInferReqTests : public ExecutableNetworkTests {
protected:
    std::shared_ptr<MockIInferRequestInternal> mockIInferReq_p;

    virtual void TearDown() {
        ExecutableNetworkTests::TearDown();
        mockIInferReq_p.reset();
    }

    virtual void SetUp() {
        ExecutableNetworkTests::SetUp();
        mockIInferReq_p = std::make_shared<MockIInferRequestInternal>();
    }
};

TEST_F(ExecutableNetworkWithIInferReqTests, CanCreateInferRequest) {
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Return(mockIInferReq_p));
    IInferRequestInternal::Ptr actualInferReq;
    ASSERT_NO_THROW(actualInferReq = exeNetwork->CreateInferRequest());
}

TEST_F(ExecutableNetworkWithIInferReqTests, CreateInferRequestThrowsIfReturnNotOK) {
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).WillOnce(Throw(InferenceEngine::GeneralError{""}));
    ASSERT_THROW(exeNetwork->CreateInferRequest(), InferenceEngine::Exception);
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
    auto inferReqInternal = std::make_shared<MockIInferRequestInternal>();
    EXPECT_CALL(*mock_impl.get(), CreateInferRequest()).Times(1).WillRepeatedly(Return(inferReqInternal));
    IInferRequest::Ptr req;
    ASSERT_NO_THROW(exeNetwork->CreateInferRequest(req, &dsc));
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
