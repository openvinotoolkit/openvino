// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include "behavior/compiled_model/properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {
namespace behavior {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::device::id(0)},
};

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_HeteroOVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values(ov::test::utils::DEVICE_HETERO));

INSTANTIATE_TEST_SUITE_P(
    smoke_HeteroOVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigPropsTest,
                         OVClassSetDevicePriorityConfigPropsTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(multiConfigs)));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("HETERO:GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("HETERO:GPU"));

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

INSTANTIATE_TEST_SUITE_P(nightly_HETERO_OVClassCompileModelWithCorrectSecondaryPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("HETERO:GPU"),
                                            ::testing::ValuesIn(gpuCorrectConfigsWithSecondaryProperties())));
}  // namespace behavior
}  // namespace test
}  // namespace ov
