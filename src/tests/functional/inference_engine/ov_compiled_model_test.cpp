// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/core/except.hpp>
#include <openvino/runtime/compiled_model.hpp>

using namespace ::testing;
using namespace std;

TEST(ExecutableNetworkOVTests, throwsOnUninitializedExportStream) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.export_model(std::cout), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetFunction) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.get_runtime_model(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputs) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.outputs(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutput) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.output(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputTensor) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.output("tensor"), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputIndex) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.output(1), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputs) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.inputs(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInput) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.input(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputTensor) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.input("tensor"), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputIndex) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.input(1), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedSetConfig) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.set_config({{}}), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetConfig) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.get_config({}), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetMetric) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.get_metric({}), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetContext) {
    ov::runtime::CompiledModel exec;
    ASSERT_THROW(exec.get_context(), ov::Exception);
}
