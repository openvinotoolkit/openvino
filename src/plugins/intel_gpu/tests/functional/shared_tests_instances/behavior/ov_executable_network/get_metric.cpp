// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("GPU", "HETERO:GPU", "BATCH:GPU"));


const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_GPU = {
    {"GPU", std::make_pair(ov::AnyMap{}, "GPU.0")},
    {"GPU.0", std::make_pair(ov::AnyMap{}, "GPU.0")},
    {"BATCH:GPU", std::make_pair(ov::AnyMap{}, "GPU.0")}};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::ValuesIn(GetMetricTest_ExecutionDevice_GPU));
//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("GPU", "HETERO:GPU", "BATCH:GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetConfigTest,
                         OVClassCompiledModelGetConfigTest,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelSetIncorrectConfigTest,
                         OVClassCompiledModelSetIncorrectConfigTest,
                         ::testing::Values("GPU"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(smoke_OVCompiledModelIncorrectDevice, OVCompiledModelIncorrectDevice, ::testing::Values("GPU"));

const std::vector<ov::AnyMap> incorrect_device_priorities_properties = {{ov::device::priorities("NONE")},
                                                                        {ov::device::priorities("NONE", "GPU")},
                                                                        {ov::device::priorities("-", "GPU")},
                                                                        {ov::device::priorities("-NONE", "CPU")},
                                                                        {ov::device::priorities("-CPU", "-NONE")},
                                                                        {ov::device::priorities("-NONE", "-NONE")}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorIncorrectPropertiesTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(incorrect_device_priorities_properties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> gpuCorrectConfigs = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::hint::allow_auto_batching(false)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::hint::allow_auto_batching(true)}};

auto gpuCorrectConfigsWithSecondaryProperties = []() {
    return std::vector<ov::AnyMap>{
        {ov::device::properties(ov::test::utils::DEVICE_GPU,
                                ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE),
                                ov::hint::inference_precision(ov::element::f32))},
        {ov::device::properties(ov::test::utils::DEVICE_GPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::allow_auto_batching(false))},
        {ov::device::properties(ov::test::utils::DEVICE_GPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::allow_auto_batching(false)),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                ov::hint::allow_auto_batching(false))}};
};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompileModelWithCorrectPropertiesAutoBatchingTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(gpuCorrectConfigs)));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompileModelWithCorrectSecondaryPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(gpuCorrectConfigsWithSecondaryProperties())),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_HETERO_OVClassCompileModelWithCorrectSecondaryPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("HETERO:GPU"),
                                            ::testing::ValuesIn(gpuCorrectConfigsWithSecondaryProperties())));

const std::vector<ov::AnyMap> batchCorrectConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_Batch_OVClassCompileModelWithCorrectPropertiesAutoBatchingTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("BATCH:GPU"), ::testing::ValuesIn(batchCorrectConfigs)));
}  // namespace
