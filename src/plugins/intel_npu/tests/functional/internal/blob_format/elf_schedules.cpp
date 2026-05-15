// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elf_schedules.hpp"

#include "common_test_utils/test_constants.hpp"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> emptyConfig = {{}};

const std::vector<ov::AnyMap> weightsSeparationConfig = {
    {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::PLUGIN},
     {ov::intel_npu::weightless_blob.name(), true}}};

const std::vector<ov::AnyMap> nonWeightsSeparationConfig = {
    {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::PLUGIN},
     {ov::intel_npu::weightless_blob.name(), false}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ELFSchedulesSections,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfig)),
                         ELFSchedulesSections::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTestWeightsSeparation,
                         ELFSchedulesSections,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(weightsSeparationConfig)),
                         ELFSchedulesSections::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ELFSchedulesWeightsSeparation,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(weightsSeparationConfig)),
                         ELFSchedulesSections::getTestCaseName);

const std::vector<ov::AnyMap> driverConfig = {
    {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::DRIVER}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ELFSchedulesNoInits,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(nonWeightsSeparationConfig)),
                         ELFSchedulesSections::getTestCaseName);

}  // namespace
