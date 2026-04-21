// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/properties.hpp>

#include "common/functions.hpp"
#include "common/utils.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

class CompatibilityStringTest : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {
protected:
    std::shared_ptr<ov::Model> create_dummy_model() {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 224, 224});
        auto result = std::make_shared<ov::op::v0::Result>(param);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "DummyModel");
    }
};

TEST_P(CompatibilityStringTest, CompileAndCheckRequirements) {
    ov::Core core;
    if (std::find(core.get_available_devices().begin(), core.get_available_devices().end(), "NPU") ==
        core.get_available_devices().end()) {
        std::cout << "[ DEBUG ] NPU device not found. Skipping test." << std::endl;
        GTEST_SKIP() << "NPU device not found";
    }

    auto compiler_type = std::get<0>(GetParam());
    auto serializer_version = std::get<1>(GetParam());

    std::cout << "[ DEBUG ] Config: NPU_COMPILER_TYPE=" << compiler_type
              << ", NPU_MODEL_SERIALIZER_VERSION=" << serializer_version << std::endl;

    auto model = create_dummy_model();

    ov::AnyMap config = {{"NPU_COMPILER_TYPE", compiler_type}, {"NPU_MODEL_SERIALIZER_VERSION", serializer_version}};

    ov::CompiledModel compiled_model;
    try {
        compiled_model = core.compile_model(model, "NPU", config);
        std::cout << "[ DEBUG ] Model compiled successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[ DEBUG ] Compilation exception: " << e.what() << std::endl;
        GTEST_SKIP() << "Compilation failed for this test condition";
    }

    ov::Tensor requirements_tensor;
    try {
        requirements_tensor = compiled_model.get_property(ov::runtime_requirements);
    } catch (const std::exception& e) {
        // As defined in the CIP workflow, old driver without string support will throw
        std::cout << "[ DEBUG ] get_property(runtime_requirements) exception: " << e.what() << std::endl;
        GTEST_SKIP() << "runtime_requirements is not supported by current driver implementation";
    }

    EXPECT_GT(requirements_tensor.get_byte_size(), 0);

    std::string comp_str(static_cast<const char*>(requirements_tensor.data()), requirements_tensor.get_byte_size());
    std::cout << "[ DEBUG ] requirements_tensor size: " << requirements_tensor.get_byte_size() << " bytes" << std::endl;
    std::cout << "[ DEBUG ] requirements_tensor content: " << comp_str << std::endl;

    auto check_result =
        core.get_property("NPU", ov::runtime_requirements_met, ov::runtime_requirements(requirements_tensor));

    std::cout << "[ DEBUG ] check_result enum value: " << static_cast<int>(check_result) << std::endl;

    bool passed = (check_result == ov::RuntimeRequirementCheckResult::COMPATIBILITY_PASSED) ||
                  (check_result == ov::RuntimeRequirementCheckResult::PARTIAL_CHECK_PASSED);

    EXPECT_TRUE(passed) << "Check result returned: " << static_cast<int>(check_result);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
