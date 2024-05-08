// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request_run.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "npu_private_properties.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> pluginCompilerConfigsInferRequestRunTests = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
         ov::log::level(ov::log::Level::INFO)}};

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigsInferRequestRunTests = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER), ov::log::level(ov::log::Level::INFO)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, InferRequestRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_Driver, InferRequestRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);

const std::vector<ov::AnyMap> batchingConfigs = {
        {ov::log::level(ov::log::Level::WARNING), ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::COMPILER)},
        {ov::log::level(ov::log::Level::WARNING), ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::AUTO)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_Driver, BatchingRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(batchingConfigs)),
                         InferRequestRunTests::getTestCaseName);
