// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

namespace {
using ov::test::behavior::OVClassCompiledModelGetPropertyTest;
using ov::test::behavior::OVClassCompiledModelGetPropertyTest_EXEC_DEVICES;
using ov::test::behavior::OVClassCompiledModelGetIncorrectPropertyTest;
using ov::test::behavior::OVClassCompiledModelGetConfigTest;
using ov::test::behavior::OVClassCompiledModelSetIncorrectConfigTest;
using ov::test::behavior::OVClassCompiledModelPropertiesIncorrectTests;
using ov::test::behavior::OVClassCompileModelWithCorrectPropertiesTest;
using ov::test::behavior::OVCompiledModelIncorrectDevice;


//
// Executable Network GetMetric
//
INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("GPU"));


const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_GPU = {
    {"GPU", std::make_pair(ov::AnyMap{}, "GPU.0")},
    {"GPU.0", std::make_pair(ov::AnyMap{}, "GPU.0")}};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::ValuesIn(GetMetricTest_ExecutionDevice_GPU));


//
// Executable Network GetConfig / SetConfig
//
INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetConfigTest,
                         OVClassCompiledModelGetConfigTest,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelSetIncorrectConfigTest,
                         OVClassCompiledModelSetIncorrectConfigTest,
                         ::testing::Values("GPU"));


// OV Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_OVCompiledModelIncorrectDevice, OVCompiledModelIncorrectDevice, ::testing::Values("GPU"));

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
}  // namespace
