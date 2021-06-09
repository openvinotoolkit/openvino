// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cpp/ie_executable_network.hpp>
#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

IE_SUPPRESS_DEPRECATED_START

TEST(ExecutableNetworkTests, throwsOnUninitialized) {
    std::shared_ptr<IExecutableNetwork> ptr;
    ASSERT_THROW(ExecutableNetwork req(ptr), InferenceEngine::NotAllocated);
}

TEST(ExecutableNetworkTests, nothrowOnInitialized) {
    std::shared_ptr<IExecutableNetwork> ptr = std::make_shared<MockIExecutableNetwork>();
    ASSERT_NO_THROW(ExecutableNetwork req(ptr));
}

IE_SUPPRESS_DEPRECATED_END

TEST(ExecutableNetworkTests, throwsOnUninitializedGetOutputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetOutputsInfo(), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetInputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetInputsInfo(), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedExport) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::string()), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedExportStream) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::cout), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetExecGraphInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetExecGraphInfo(), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedQueryState) {
    IE_SUPPRESS_DEPRECATED_START
    ExecutableNetwork exec;
    ASSERT_THROW(exec.QueryState(), InferenceEngine::Exception);
    IE_SUPPRESS_DEPRECATED_END
}

TEST(ExecutableNetworkTests, throwsOnUninitializedSetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.SetConfig({{}}), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetConfig({}), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetMetric) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetMetric({}), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetContext) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetContext(), InferenceEngine::Exception);
}