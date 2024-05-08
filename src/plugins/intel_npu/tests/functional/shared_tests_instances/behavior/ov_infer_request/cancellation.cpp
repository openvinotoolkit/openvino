//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_infer_request/cancellation.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "overload/ov_infer_request/cancellation.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> pluginCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}};

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCancellationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCancellationTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestCancellationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestCancellationTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);
}  // namespace
