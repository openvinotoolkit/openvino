// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include <openvino/runtime/auto/properties.hpp>

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCommon,
                         OVBasicPropertiesTestsP,
                         ::testing::Values(std::make_pair("openvino_intel_cpu_plugin", "CPU")));

const std::vector<ov::AnyMap> cpu_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(cpu_properties)),
                         OVPropertiesTests::getTestCaseName);

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
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_HeteroOVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values("HETERO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
    smoke_HeteroOVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values("HETERO"),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

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

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_CPU)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigPropsTest,
                         OVClassSetDevicePriorityConfigPropsTest,
                         ::testing::Combine(::testing::Values("HETERO"),
                                            ::testing::ValuesIn(multiConfigs)));

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


// IE Class load and check network with ov::device::properties
INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"), ::testing::ValuesIn(configsDeviceProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassCompileModelAndCheckWithSecondaryPropertiesDoubleTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsDevicePropertiesDouble)));

}  // namespace
