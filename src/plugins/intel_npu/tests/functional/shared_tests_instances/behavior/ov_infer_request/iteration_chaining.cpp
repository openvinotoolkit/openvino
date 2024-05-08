//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_infer_request/iteration_chaining.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "npu_private_properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configs = {
        {{ov::hint::inference_precision.name(), ov::element::f32},
         {ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::MLIR},
         {ov::intel_npu::platform.name(), ov::test::utils::getTestsPlatformCompilerInPlugin()}}};

const std::vector<ov::AnyMap> configsDriver = {
        {{ov::hint::inference_precision.name(), ov::element::f32},
         {ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::DRIVER}}};

const std::vector<ov::AnyMap> heteroConfigs = {
        {{ov::hint::inference_precision.name(), ov::element::f32},
         {ov::device::priorities(ov::test::utils::DEVICE_NPU)},
         {ov::device::properties(ov::test::utils::DEVICE_NPU,
                                 {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::MLIR},
                                  {ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}})}}};

const std::vector<ov::AnyMap> heteroConfigsDriver = {
        {{ov::hint::inference_precision.name(), ov::element::f32},
         {ov::device::priorities(ov::test::utils::DEVICE_NPU)},
         {ov::device::properties(ov::test::utils::DEVICE_NPU,
                                 {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::DRIVER}})}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVIterationChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVIterationChaining>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVIterationChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(heteroConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVIterationChaining>);
}  // namespace
