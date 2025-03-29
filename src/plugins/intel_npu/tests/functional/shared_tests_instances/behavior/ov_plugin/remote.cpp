// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/remote.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"

using namespace ov::test;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<std::pair<ov::AnyMap, ov::AnyMap>> generate_remote_params{
    {{}, {}},
    {{}, {ov::intel_npu::mem_type(ov::intel_npu::MemType::L0_INTERNAL_BUF)}},
    {{},
     {ov::intel_npu::mem_type(ov::intel_npu::MemType::L0_INTERNAL_BUF),
      ov::intel_npu::tensor_type(ov::intel_npu::TensorType::BINDED)}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVRemoteTest,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::Values(::ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(generate_remote_params)),
                         (ov::test::utils::appendPlatformTypeTestName<OVRemoteTest>));
}  // namespace
