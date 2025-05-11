// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <openvino/runtime/compiled_model.hpp>

using namespace ::testing;
using namespace std;

TEST(ExecutableNetworkOVTests, throwsOnUninitializedExportStream) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.export_model(std::cout), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetFunction) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.get_runtime_model(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputs) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.outputs(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutput) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.output(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputTensor) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.output("tensor"), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedOutputIndex) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.output(1), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputs) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.inputs(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInput) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.input(), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputTensor) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.input("tensor"), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedInputIndex) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.input(1), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedSetConfig) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.set_property({{}}), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetMetric) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.get_property({}), ov::Exception);
}

TEST(ExecutableNetworkOVTests, throwsOnUninitializedGetContext) {
    ov::CompiledModel exec;
    ASSERT_THROW(exec.get_context(), ov::Exception);
}
