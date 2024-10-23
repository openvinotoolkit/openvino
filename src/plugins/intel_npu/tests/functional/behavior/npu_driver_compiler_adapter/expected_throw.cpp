// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expected_throw.hpp"

#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {{{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DriverCompilerAdapterExpectedThrowNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         DriverCompilerAdapterExpectedThrowNPU::getTestCaseName);
