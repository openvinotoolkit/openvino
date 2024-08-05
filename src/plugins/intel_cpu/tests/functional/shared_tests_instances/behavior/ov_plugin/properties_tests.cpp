// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/auto/properties.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCommon,
                         OVBasicPropertiesTestsP,
                         ::testing::Values(std::make_pair("openvino_intel_cpu_plugin", "CPU")));

auto cpu_properties = []() -> std::vector<ov::AnyMap> {
    std::vector<ov::AnyMap> properties = {
        {},
        {ov::hint::enable_cpu_pinning(true)},
        {ov::hint::enable_cpu_pinning(false)},
        {ov::enable_profiling(true)},
        {ov::enable_profiling(false)},
        {ov::internal::exclusive_async_requests(true)},
        {ov::internal::exclusive_async_requests(false)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}, {ov::hint::num_requests(1)}},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::num_streams(ov::streams::AUTO)},
        {ov::num_streams(8)},
        // check that hints doesn't override customer value (now for streams and later for other config opts)
        {{ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}, {ov::hint::num_requests(3)}},
        {{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}, {ov::hint::num_requests(3)}},
    };

    auto numa_nodes = ov::get_available_numa_nodes();
    if (numa_nodes.size() > 1) {
        properties.push_back({ov::num_streams(ov::streams::NUMA)});
    }
    return properties;
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(cpu_properties())),
                         OVPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> cpu_inproperties = {
    {{ov::hint::performance_mode.name(), "DOESN'T EXIST"}},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY), {ov::hint::num_requests(-1)}},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     {ov::hint::num_requests.name(), "should be int"}},
    {{ov::num_streams.name(), "OFF"}},
    {{ov::hint::enable_cpu_pinning.name(), "OFF"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(cpu_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> cpu_setcore_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::num_requests(2),
     ov::enable_profiling(false)}};
const std::vector<ov::AnyMap> cpu_compileModel_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::num_requests(10),
     ov::enable_profiling(true)}};

INSTANTIATE_TEST_SUITE_P(smoke_cpuCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(cpu_setcore_properties),
                                            ::testing::ValuesIn(cpu_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_CPU));

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
    smoke_OVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVGetAvailableDevicesPropsTest,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values("CPU"));

const std::vector<ov::AnyMap> configsDeviceProperties = {
    {ov::device::properties("CPU", ov::num_streams(2))},
    {ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(2)}}})}};

const std::vector<ov::AnyMap> configsDevicePropertiesDouble = {
    {ov::device::properties("CPU", ov::num_streams(2)), ov::num_streams(5)},
    {ov::device::properties("CPU", ov::num_streams(2)),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(7)}}}),
     ov::num_streams(5)},
    {ov::device::properties("CPU", ov::num_streams(2)), ov::device::properties("CPU", ov::num_streams(5))},
    {ov::device::properties("CPU", ov::num_streams(1)),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(5)}}})},
    {ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(1)}}}),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(5)}}})}};







// OV Class load and check network with ov::device::properties
INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"), ::testing::ValuesIn(configsDeviceProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassCompileModelAndCheckWithSecondaryPropertiesDoubleTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsDevicePropertiesDouble)));

}  // namespace
