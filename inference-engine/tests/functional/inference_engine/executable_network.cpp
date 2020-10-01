// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cpp/ie_executable_network.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

TEST(ExecutableNetworkTests, smoke_throwsOnInitWithNull) {
    std::shared_ptr<IExecutableNetwork> nlptr = nullptr;
    ASSERT_THROW(ExecutableNetwork exec(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedGetOutputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetOutputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedGetInputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetInputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedExport) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::string()), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedExportStream) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::cout), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_nothrowsOnUninitializedCast) {
    ExecutableNetwork exec;
    ASSERT_NO_THROW(auto &enet = static_cast<IExecutableNetwork::Ptr &>(exec));
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedGetExecGraphInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetExecGraphInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedQueryState) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.QueryState(), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedSetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.SetConfig({{}}), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedGetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetConfig({}), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedGetMetric) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetMetric({}), InferenceEngine::details::InferenceEngineException);
}

TEST(ExecutableNetworkTests, smoke_throwsOnUninitializedGetContext) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetContext(), InferenceEngine::details::InferenceEngineException);
}