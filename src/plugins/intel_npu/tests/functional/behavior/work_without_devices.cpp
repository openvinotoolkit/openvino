// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/work_without_devices.hpp"

#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace {

const std::vector<ov::AnyMap> configs = {
    {{"NPU_COMPILER_TYPE", "PLUGIN"}, {"NPU_PLATFORM", "NPU4000"}, {"NPU_CREATE_EXECUTOR", "0"}},
    {{"NPU_COMPILER_TYPE", "PLUGIN"}, {"NPU_PLATFORM", "NPU5010"}, {"NPU_CREATE_EXECUTOR", "0"}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         TestOfflineCompilationNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         TestOfflineCompilationNPU::getTestCaseName);

}  // namespace
