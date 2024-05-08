//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <base/ov_behavior_test_utils.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "npu_private_properties.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

using CompileWithDummy_NPU3720 = ov::test::behavior::OVInferRequestTests;

std::shared_ptr<ov::Model> buildSingleLayerClampNetwork() {  // Clamp is not supported in SW
    ov::Shape inputShape = {1, 3, 4, 3};
    ov::element::Type netPrecision = ov::element::f32;

    const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape{inputShape})};

    const auto clamp = std::make_shared<ov::op::v0::Clamp>(params.at(0), 0., 1.);

    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(clamp)};

    auto ov_model = std::make_shared<ov::Model>(results, params, "clamp");

    return ov_model;
}

namespace {

TEST_P(CompileWithDummy_NPU3720, CompilationForSpecificPlatform) {
    if (getBackendName(*core) == "LEVEL0") {
        GTEST_SKIP() << "Skip due to failure on device";
    }
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& ov_model = buildSingleLayerClampNetwork();
        OV_ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_model, target_device, configuration));
    }
}

const std::vector<ov::AnyMap> pluginCompilerConfigs = {
        {{ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720)},
         {ov::intel_npu::compilation_mode_params("dummy-op-replacement=true")}}};
// Must be successfully compiled with dummy-op-replacement=true

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {{ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720)},
         {ov::intel_npu::compilation_mode_params("dummy-op-replacement=true")},
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BehaviorTest_Dummy, CompileWithDummy_NPU3720,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         CompileWithDummy_NPU3720::getTestCaseName);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_precommit_BehaviorTest_Dummy_Driver, CompileWithDummy_NPU3720,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         CompileWithDummy_NPU3720::getTestCaseName);
}  // namespace
