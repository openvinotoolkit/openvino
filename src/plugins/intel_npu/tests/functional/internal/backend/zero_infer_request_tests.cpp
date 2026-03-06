// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/backend/zero_infer_request_tests.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configs = {{{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                                           ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::PLUGIN)}},
                                         {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                                           ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::COMPILER)}},
                                         {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                                           ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::PLUGIN)}},
                                         {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                                           ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::COMPILER)}}};
const std::vector<ov::element::Type> specialTensorDataTypes = {ov::element::boolean};

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTest,
    ZeroInferRequestTests,
    ::testing::Combine(
        ::testing::Values(ov::test::utils::DEVICE_NPU),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(specialTensorDataTypes),
        ::testing::ValuesIn(std::vector<bool>{true, false}),  // with warmup infer
        ::testing::ValuesIn(std::vector<bool>{true, false}),  // with reset infer request
        // Simulate PV driver having 1.5 graph extension version and no mutable command list extension version
        ::testing::ValuesIn(std::vector<std::pair<uint32_t, uint32_t>>{
            {ZE_GRAPH_EXT_VERSION_1_5, ZE_MAKE_VERSION(0, 0)},
            {TARGET_ZE_GRAPH_NPU_EXT_VERSION, TARGET_ZE_MUTABLE_COMMAND_LIST_EXT_VERSION}})),
    ZeroInferRequestTests::getTestCaseName);
