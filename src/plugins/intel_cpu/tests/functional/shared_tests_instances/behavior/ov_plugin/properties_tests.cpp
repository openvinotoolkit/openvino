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

const std::vector<ov::AnyMap> multi_Auto_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU), ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU), ov::intel_auto::device_bind_buffer("YES")},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU), ov::intel_auto::device_bind_buffer("NO")},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU), ov::intel_auto::enable_startup_fallback("YES")},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU), ov::intel_auto::enable_startup_fallback("NO")}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiBehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_Auto_properties)),
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

const std::vector<ov::AnyMap> multi_setcore_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::HIGH)}};
const std::vector<ov::AnyMap> multi_compileModel_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)}};

INSTANTIATE_TEST_SUITE_P(smoke_MultiCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_setcore_properties),
                                            ::testing::ValuesIn(multi_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_setcore_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
};
const std::vector<ov::AnyMap> auto_compileModel_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)}};
INSTANTIATE_TEST_SUITE_P(smoke_AutoCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_setcore_properties),
                                            ::testing::ValuesIn(auto_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {{ov::enable_profiling(false)},
                                                    {ov::log::level("LOG_NONE")},
                                                    {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
                                                    {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
                                                    {ov::intel_auto::device_bind_buffer(false)},
                                                    {ov::intel_auto::enable_startup_fallback(true)},
                                                    {ov::device::priorities("")}};
INSTANTIATE_TEST_SUITE_P(smoke_AutoBehaviorTests,
                         OVPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(default_properties)),
                         OVPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_AUTO));

const std::vector<ov::AnyMap> auto_multi_incorrect_device_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::num_streams(4),
     ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::num_streams(4),
     ov::device::properties("CPU", ov::num_streams(4), ov::enable_profiling(true))}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsThrow,
                         OVSetUnsupportPropCompileModelWithoutConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI,
                                                              ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(auto_multi_incorrect_device_properties)),
                         OVSetUnsupportPropCompileModelWithoutConfigTests::getTestCaseName);

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiHeteroOVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values("MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
    smoke_AutoMultiHeteroOVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values("MULTI", "HETERO", "AUTO"),
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

INSTANTIATE_TEST_SUITE_P(
    OVCheckSetSupportedRWMandatoryMetricsPropsTests,
    OVCheckSetSupportedRWMetricsPropsTests,
    ::testing::Combine(::testing::Values("MULTI:CPU", "AUTO:CPU"),
                       ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWMandatoryPropertiesValues(
                           {ov::hint::model_priority.name(), ov::log::level.name()}))),
    OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    OVCheckSetSupportedRWOptionalMetricsPropsTests,
    OVCheckSetSupportedRWMetricsPropsTests,
    ::testing::Combine(::testing::Values("MULTI:CPU", "AUTO:CPU"),
                       ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWOptionalPropertiesValues(
                           {ov::hint::enable_hyper_threading.name(),
                            ov::hint::enable_cpu_pinning.name(),
                            ov::hint::scheduling_core_type.name()}))),
    OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_CPU)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigPropsTest,
                         OVClassSetDevicePriorityConfigPropsTest,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO", "HETERO"),
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
