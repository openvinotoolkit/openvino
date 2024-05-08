//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "overload/compile_and_infer.hpp"
#include <npu_private_properties.hpp>
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompileAndInferRequest,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequest>);

}  // namespace
