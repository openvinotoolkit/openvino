// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compatibility_string.hpp"

namespace ov {
namespace test {
namespace behavior {

TEST_F(CompatibilityStringTest, CompileAndCheckRequirements) {
    ov::Core core;
    if (std::find(core.get_available_devices().begin(), core.get_available_devices().end(), "NPU") ==
        core.get_available_devices().end()) {
        GTEST_SKIP() << "NPU device not found";
    }

    auto model = create_dummy_model();

    auto compiled_model = core.compile_model(model, "NPU");
    ov::Tensor requirements_tensor = compiled_model.get_property(ov::runtime_requirements);

    EXPECT_GT(requirements_tensor.get_byte_size(), 0);

    auto check_result =
        core.get_property("NPU", ov::runtime_requirements_met, ov::runtime_requirements(requirements_tensor));

    bool passed = (check_result == ov::RuntimeRequirementCheckResult::COMPATIBILITY_PASSED) ||
                  (check_result == ov::RuntimeRequirementCheckResult::PARTIAL_CHECK_PASSED);

    EXPECT_TRUE(passed) << "Check result returned: " << static_cast<int>(check_result);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
