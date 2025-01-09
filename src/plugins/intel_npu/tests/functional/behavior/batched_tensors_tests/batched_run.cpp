// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/batched_tensors_tests/batched_run.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> batchedConfigs = {{ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::PLUGIN)},
                                                {ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::COMPILER)},
                                                {ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::AUTO)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         BatchedTensorsRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(batchedConfigs)),
                         BatchedTensorsRunTests::getTestCaseName);
