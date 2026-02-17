// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::intel_npu;

using OfflineCompilationUnitTests = ::testing::Test;


TEST_F(OfflineCompilationUnitTests, wrongDriverPath) {
    ov::Core core;

    ov::AnyMap config;
    config[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::PLUGIN;
    core.set_property("NPU", config);

    std::shared_ptr<ov::Model> model = ov::test::utils::make_multi_single_conv();

    ov::CompiledModel compiled_model;

    EXPECT_ANY_THROW(compiled_model = core.compile_model(model, "NPU", {}));
}
