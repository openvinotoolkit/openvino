// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_driver_compiler_adapter_downgrade_interpolate11.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {
        {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, DriverCompilerAdapterDowngradeInterpolate11TestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         DriverCompilerAdapterDowngradeInterpolate11TestNPU::getTestCaseName);
