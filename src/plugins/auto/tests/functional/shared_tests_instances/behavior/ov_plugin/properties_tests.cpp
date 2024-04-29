// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include "openvino/runtime/auto/properties.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI,
                                                              ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> multi_Auto_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::intel_auto::device_bind_buffer("YES")},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::intel_auto::device_bind_buffer("NO")},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::intel_auto::enable_startup_fallback("YES")},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::intel_auto::enable_startup_fallback("NO")},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(false)}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiBehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_Auto_properties)),
                         OVPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> multi_setcore_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::HIGH)}};

const std::vector<ov::AnyMap> multi_compileModel_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(false)}};

INSTANTIATE_TEST_SUITE_P(smoke_MultiCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_setcore_properties),
                                            ::testing::ValuesIn(multi_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_setcore_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::HIGH)}};

const std::vector<ov::AnyMap> auto_compileModel_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(false)}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_setcore_properties),
                                            ::testing::ValuesIn(auto_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
    {ov::enable_profiling(false)},
    {ov::log::level("LOG_NONE")},
    {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::intel_auto::device_bind_buffer(false)},
    {ov::intel_auto::enable_startup_fallback(true)},
    {ov::intel_auto::schedule_policy(ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY)},
    {ov::device::priorities("")}};
INSTANTIATE_TEST_SUITE_P(smoke_AutoBehaviorTests,
                         OVPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(default_properties)),
                         OVPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE, ov::test::utils::DEVICE_AUTO));

const std::vector<ov::AnyMap> auto_multi_incorrect_device_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::num_streams(4),
     ov::device::properties("TEMPLATE", ov::num_streams(4))},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::num_streams(4),
     ov::device::properties("TEMPLATE", ov::num_streams(4), ov::enable_profiling(true))}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsThrow,
                         OVSetUnsupportPropCompileModelWithoutConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_multi_incorrect_device_properties)),
                         OVSetUnsupportPropCompileModelWithoutConfigTests::getTestCaseName);

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_AutoOVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
    smoke_AutoOVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_OVCheckSetSupportedRWMetricsPropsTests,
    OVCheckSetSupportedRWMetricsPropsTests,
    ::testing::Combine(::testing::Values("MULTI:TEMPLATE", "AUTO:TEMPLATE"),
                       ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWOptionalPropertiesValues(
                           {ov::log::level.name()}))),
    OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigPropsTest,
                         OVClassSetDevicePriorityConfigPropsTest,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"), ::testing::ValuesIn(multiConfigs)));

const std::vector<ov::AnyMap> auto_properties = {{ov::device::priorities("TEMPLATE")},
                                                 {ov::device::priorities("TEMPLATE(1)")}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultiBehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_MultiAutoOVCheckSetSupportedRWMetricsPropsTests,
    OVCheckSetSupportedRWMetricsPropsTests,
    ::testing::Combine(::testing::Values("MULTI:TEMPLATE", "AUTO:TEMPLATE"),
                       ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWOptionalPropertiesValues(
                           {ov::log::level.name()}))),
    OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);
}  // namespace
