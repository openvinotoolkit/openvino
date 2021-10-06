// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/core/except.hpp>
#include <openvino/runtime/executable_network.hpp>

using namespace ::testing;
using namespace std;

TEST(ExecutableNetworkOVTests, throwsOnUninitializedExportStream) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.export_model(std::cout), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetFunction) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_runtime_function(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetParameters) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_parameters(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetResults) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_results(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedSetConfig) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.set_config({{}}), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetConfig) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_config({}), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetMetric) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_metric({}), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetContext) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_context(), ov::Exception);
}