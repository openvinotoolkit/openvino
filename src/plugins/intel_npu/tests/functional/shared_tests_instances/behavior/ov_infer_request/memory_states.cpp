//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_infer_request/memory_states.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "npu_private_properties.hpp"

using namespace ov::test::behavior;
using namespace ov;

namespace {
const std::vector<memoryStateParams> memoryStateTestCases = {memoryStateParams(
        OVInferRequestVariableStateTest::get_network(), {"c_1-3", "r_1-3"}, ov::test::utils::DEVICE_NPU,
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())})};

const std::vector<memoryStateParams> memoryStateTestCases_Driver = {memoryStateParams(
        OVInferRequestVariableStateTest::get_network(), {"c_1-3", "r_1-3"}, ov::test::utils::DEVICE_NPU,
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)})};

const std::vector<memoryStateParams> memoryStateHeteroTestCases = {memoryStateParams(
        OVInferRequestVariableStateTest::get_network(), {"c_1-3", "r_1-3"}, ov::test::utils::DEVICE_HETERO,
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
                                 ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())})})};

const std::vector<memoryStateParams> memoryStateHeteroTestCases_Driver = {memoryStateParams(
        OVInferRequestVariableStateTest::get_network(), {"c_1-3", "r_1-3"}, ov::test::utils::DEVICE_HETERO,
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)})})};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_VariableState, OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestVariableStateTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_VariableState_Driver, OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases_Driver),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestVariableStateTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests_VariableState, OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateHeteroTestCases),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestVariableStateTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests_VariableState_Driver, OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateHeteroTestCases_Driver),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestVariableStateTest>);

}  // namespace
