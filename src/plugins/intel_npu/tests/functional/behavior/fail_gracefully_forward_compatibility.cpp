// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/fail_gracefully_forward_compatibility.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"

using namespace ov::test::behavior;

bool UnsupportedTestOperation::visit_attributes(AttributeVisitor& /*visitor*/) {
    return true;
}

const std::vector<ov::AnyMap> mlirCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}};
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, FailGracefullyTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(mlirCompilerConfigs)),
                         FailGracefullyTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_Driver, FailGracefullyTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         FailGracefullyTest::getTestCaseName);
