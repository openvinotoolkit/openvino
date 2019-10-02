// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"

#include <ie_version.hpp>
#include "cpp_interfaces/base/ie_plugin_base.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class ExecutableNetworkBaseTests : public ::testing::Test {
protected:
    shared_ptr<MockIExecutableNetworkInternal> mock_impl;
    shared_ptr<IExecutableNetwork> exeNetwork;
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
    EXPECT_CALL(*mock_impl.get(), CreateInferRequest(Ref(req))).Times(1);
    ASSERT_EQ(OK, exeNetwork->CreateInferRequest(req, &dsc));
}

TEST_F(ExecutableNetworkBaseTests, canReportErrorInCreateInferRequest) {
    EXPECT_CALL(*mock_impl.get(), CreateInferRequest(_)).WillOnce(Throw(std::runtime_error("compare")));
    IInferRequest::Ptr req;
    ASSERT_NE(exeNetwork->CreateInferRequest(req, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(ExecutableNetworkBaseTests, canCatchUnknownErrorInCreateInferRequest) {
    EXPECT_CALL(*mock_impl.get(), CreateInferRequest(_)).WillOnce(Throw(5));
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

// GetMappedTopology
TEST_F(ExecutableNetworkBaseTests, canForwardGetMappedTopology) {
    std::map<std::string, std::vector<PrimitiveInfo::Ptr>> deployedTopology;
    EXPECT_CALL(*mock_impl.get(), GetMappedTopology(Ref(deployedTopology))).Times(1);
    ASSERT_EQ(OK, exeNetwork->GetMappedTopology(deployedTopology, &dsc));
}

TEST_F(ExecutableNetworkBaseTests, canReportErrorInCreateInferRequestGetMappedTopology) {
    EXPECT_CALL(*mock_impl.get(), GetMappedTopology(_)).WillOnce(Throw(std::runtime_error("compare")));
    std::map<std::string, std::vector<PrimitiveInfo::Ptr>> deployedTopology;
    ASSERT_NE(exeNetwork->GetMappedTopology(deployedTopology, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(ExecutableNetworkBaseTests, canCatchUnknownErrorInGetMappedTopology) {
    EXPECT_CALL(*mock_impl.get(), GetMappedTopology(_)).WillOnce(Throw(5));
    std::map<std::string, std::vector<PrimitiveInfo::Ptr>> deployedTopology;
    ASSERT_EQ(UNEXPECTED, exeNetwork->GetMappedTopology(deployedTopology, nullptr));
}
