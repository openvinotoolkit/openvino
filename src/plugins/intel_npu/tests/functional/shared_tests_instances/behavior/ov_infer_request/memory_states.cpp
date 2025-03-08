// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/memory_states.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;
using namespace ov;

namespace {
const std::vector<memoryStateParams> memoryStateTestCases = {
    memoryStateParams(OVInferRequestVariableStateTest::get_network(),
                      {"c_1-3", "r_1-3"},
                      ov::test::utils::DEVICE_NPU,
                      {})};

const std::vector<memoryStateParams> memoryStateHeteroTestCases = {memoryStateParams(
    OVInferRequestVariableStateTest::get_network(),
    {"c_1-3", "r_1-3"},
    ov::test::utils::DEVICE_HETERO,
    {ov::device::priorities(ov::test::utils::DEVICE_NPU), ov::device::properties(ov::test::utils::DEVICE_NPU, {})})};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_VariableState,
                         OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestVariableStateTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests_VariableState,
                         OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateHeteroTestCases),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestVariableStateTest>);

}  // namespace
