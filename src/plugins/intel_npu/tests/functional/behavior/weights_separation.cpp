// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/weights_separation.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> emptyConfig = {{}};
const std::vector<ov::AnyMap> cipConfig = {
    {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
     ov::intel_npu::platform(ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU))}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         WeightsSeparationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfig)),
                         WeightsSeparationTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         WeightsSeparationOneShotTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(cipConfig)),
                         WeightsSeparationOneShotTests::getTestCaseName);
