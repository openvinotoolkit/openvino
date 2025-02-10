// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request_run.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configsInferRequestRunTests = {{}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         InferRequestRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         RandomTensorOverZeroTensorRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         RunSeqTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         SetShapeInferRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);

const std::vector<ov::AnyMap> batchingConfigs = {
    {ov::log::level(ov::log::Level::WARNING), ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::PLUGIN)},
    {ov::log::level(ov::log::Level::WARNING), ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::COMPILER)},
    {ov::log::level(ov::log::Level::WARNING), ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::AUTO)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         BatchingRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(batchingConfigs)),
                         InferRequestRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         BatchingRunSeqTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(batchingConfigs)),
                         InferRequestRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         ROITensorInference,
                         ::testing::Combine(::testing::Values(tensor_roi::roi_nchw()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestInferenceTests>);
