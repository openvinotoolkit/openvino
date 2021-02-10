// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cpp/ie_executable_network.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

TEST(ExecutableNetworkTests, throwsOnInitWithNull) {
    std::shared_ptr<IExecutableNetwork> nlptr = nullptr;
    ASSERT_THROW(ExecutableNetwork exec(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetOutputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetOutputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetInputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetInputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedExport) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::string()), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedExportStream) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::cout), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, nothrowsOnUninitializedCast) {
    ExecutableNetwork exec;
    ASSERT_NO_THROW((void)static_cast<IExecutableNetwork::Ptr &>(exec));
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetExecGraphInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetExecGraphInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedQueryState) {
    IE_SUPPRESS_DEPRECATED_START
    ExecutableNetwork exec;
    ASSERT_THROW(exec.QueryState(), InferenceEngine::details::InferenceEngineException);
    IE_SUPPRESS_DEPRECATED_END
}

TEST(ExecutableNetworkTests, throwsOnUninitializedSetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.SetConfig({{}}), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetConfig({}), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetMetric) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetMetric({}), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetContext) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetContext(), InferenceEngine::details::InferenceEngineException);
}