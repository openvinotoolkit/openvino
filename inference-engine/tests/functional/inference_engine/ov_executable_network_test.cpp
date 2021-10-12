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
    ASSERT_THROW(exec.export(std::cout), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetFunction) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.get_runtime_function(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputs) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.outputs(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutput) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.output(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputTensor) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.output("tensor"), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputIndex) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.output(1), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputs) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.inputs(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInput) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.input(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputTensor) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.input("tensor"), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputIndex) {
    ov::runtime::ExecutableNetwork exec;
    ASSERT_THROW(exec.input(1), ov::Exception);
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
