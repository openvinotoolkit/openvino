// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "behavior/compiled_model/properties_hetero.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {
namespace behavior {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::device::id("0")},
};

const std::vector<ov::AnyMap> heteroConfigsWithSecondaryProperties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties(ov::test::utils::DEVICE_HETERO,
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties(ov::test::utils::DEVICE_HETERO,
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_GPU),
     ov::device::properties(ov::test::utils::DEVICE_HETERO,
                            ov::enable_profiling(false),
                            ov::device::priorities(ov::test::utils::DEVICE_CPU),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

INSTANTIATE_TEST_SUITE_P(nightly_HETERO_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(heteroConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("HETERO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("HETERO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroCompiledModelGetMetricTest,
                         OVClassHeteroCompiledModelGetMetricTest_TARGET_FALLBACK,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));
INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroCompiledModelGetMetricTest,
                         OVClassHeteroCompiledModelGetMetricTest_EXEC_DEVICES,
                         ::testing::Values("TEMPLATE.0"));
}  // namespace behavior
}  // namespace test
}  // namespace ov
