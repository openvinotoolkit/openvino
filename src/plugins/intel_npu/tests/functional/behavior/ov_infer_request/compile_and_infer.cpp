// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compile_and_infer.hpp"

#include <intel_npu/npu_private_properties.hpp>

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
                                                {ov::intel_npu::defer_weights_load(true)},
                                                {ov::intel_npu::defer_weights_load(false)}})),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequestTurbo>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVCompileAndInferRequestSerializers,
                         ::testing::Combine(::testing::Values(createModelContainingSubgraph()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                                {ov::intel_npu::use_base_model_serializer(true)},
                                                {ov::intel_npu::use_base_model_serializer(false)}})),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileAndInferRequestSerializers>);

}  // namespace
