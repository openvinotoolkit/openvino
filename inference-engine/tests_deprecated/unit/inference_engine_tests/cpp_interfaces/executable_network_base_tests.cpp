// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "cpp_interfaces/base/ie_plugin_base.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"

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

using ExecutableNetworkTests = ::testing::Test;

TEST_F(ExecutableNetworkTests, throwsOnInitWithNull) {
    std::shared_ptr<IExecutableNetwork> nlptr = nullptr;
    ASSERT_THROW(ExecutableNetwork exec(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedGetOutputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetOutputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedGetInputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetInputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedExport) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::string()), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedExportStream) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::cout), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, nothrowsOnUninitializedCast) {
    ExecutableNetwork exec;
    ASSERT_NO_THROW(auto & enet = static_cast<IExecutableNetwork::Ptr&>(exec));
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedGetExecGraphInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetExecGraphInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedQueryState) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.QueryState(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedSetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.SetConfig({{}}), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedGetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetConfig({}), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedGetMetric) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetMetric({}), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecutableNetworkTests, throwsOnUninitializedGetContext) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetContext(), InferenceEngine::details::InferenceEngineException);
}
