// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/compile_and_infer.hpp"

#include <npu_private_properties.hpp>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVCompileAndInferRequest,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequest>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVCompileAndInferRequestTurbo,
                         ::testing::Combine(::testing::Values(getConstantGraph(ov::element::f32)),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::create_executor(0)},
                                                {ov::intel_npu::create_executor(1)}})),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequestTurbo>);

}  // namespace
