//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "common_test_utils/test_constants.hpp"
#include "intel_npu/al/config/common.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> pluginCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}};

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferenceChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferenceChaining>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferenceChainingStatic,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferenceChainingStatic>);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferenceChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferenceChaining>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferenceChainingStatic,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferenceChainingStatic>);
}  // namespace
