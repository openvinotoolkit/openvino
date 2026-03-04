// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request_run.hpp"

#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> configsInferRequestRunTests = {{}};
const std::vector<ov::AnyMap> configsBooleanPrecisionInferRequestRunTests = {
    {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}},
    {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         InferRequestRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         BooleanPrecisionInferRequestRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsBooleanPrecisionInferRequestRunTests)),
                         BooleanPrecisionInferRequestRunTests::getTestCaseName);

const std::vector<ov::AnyMap> profilingConfigs{{ov::intel_npu::profiling_type(ov::intel_npu::ProfilingType::MODEL)},
                                               {ov::intel_npu::profiling_type(ov::intel_npu::ProfilingType::INFER)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ProfilingBlob,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(profilingConfigs)),
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

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         CpuVaTensorsTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests)),
                         InferRequestRunTests::getTestCaseName);

const std::vector<ov::AnyMap> batchingConfigs = {{ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::PLUGIN)},
                                                 {ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::COMPILER)},
                                                 {ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::AUTO)}};

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

const std::vector<ov::AnyMap> DynamicBatchedConfigs = {{ov::intel_npu::batch_mode(ov::intel_npu::BatchMode::PLUGIN)}};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchingTests,
                         DynamicBatchingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(DynamicBatchedConfigs)),
                         InferRequestRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         ROITensorInference,
                         ::testing::Combine(::testing::Values(tensor_roi::roi_nchw()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestInferenceTests>);

}  // namespace
