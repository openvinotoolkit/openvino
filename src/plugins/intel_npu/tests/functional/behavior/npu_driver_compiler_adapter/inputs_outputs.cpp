// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inputs_outputs.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {
    {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}},
};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         DriverCompilerAdapterInputsOutputsTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         DriverCompilerAdapterInputsOutputsTestNPU::getTestCaseName);
