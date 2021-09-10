// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/runtime/executable_network.hpp>

using namespace ::testing;
using namespace std;

TEST(ExecutableNetworkOVTests, throwsOnUninitializedExportStream) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.export_model(std::cout), InferenceEngine::NotAllocated);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetFunction) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_runtime_function(), InferenceEngine::NotAllocated);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetParameters) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_parameters(), InferenceEngine::NotAllocated);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetResults) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_results(), InferenceEngine::NotAllocated);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedSetConfig) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.set_config({{}}), InferenceEngine::NotAllocated);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetConfig) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_config({}), InferenceEngine::NotAllocated);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetMetric) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_metric({}), InferenceEngine::NotAllocated);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetContext) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_context(), InferenceEngine::NotAllocated);
}