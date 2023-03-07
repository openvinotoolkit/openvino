// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include <openvino/runtime/auto/properties.hpp>

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {

const std::vector<ov::AnyMap> cpu_properties = {
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::ValuesIn(cpu_properties)),
        OVPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> multi_Auto_properties = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::execution_mode(ov::hint::ExecutionMode::UNDEFINED)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::device_bind_buffer("YES")},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::device_bind_buffer("NO")},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::enable_startup_fallback("YES")},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::enable_startup_fallback("NO")}
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiBehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO, CommonTestUtils::DEVICE_MULTI),
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
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_CPU),
                                            ::testing::ValuesIn(cpu_setcore_properties),
                                            ::testing::ValuesIn(cpu_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> multi_setcore_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::num_requests(2),
     ov::hint::allow_auto_batching(false),
     ov::enable_profiling(false)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::num_requests(8),
     ov::hint::allow_auto_batching(true),
     ov::enable_profiling(true)}};
const std::vector<ov::AnyMap> multi_compileModel_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::num_requests(10),
     ov::hint::allow_auto_batching(true),
     ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::num_requests(2),
     ov::hint::allow_auto_batching(false),
     ov::enable_profiling(false)}};

INSTANTIATE_TEST_SUITE_P(smoke_MultiCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_setcore_properties),
                                            ::testing::ValuesIn(multi_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_setcore_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::num_requests(2),
     ov::hint::allow_auto_batching(false),
     ov::enable_profiling(false)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::num_requests(8),
     ov::hint::allow_auto_batching(true),
     ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::num_requests(10),
     ov::hint::allow_auto_batching(false),
     ov::enable_profiling(true)},
};
const std::vector<ov::AnyMap> auto_compileModel_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::num_requests(8),
     ov::hint::allow_auto_batching(true),
     ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::num_requests(10),
     ov::hint::allow_auto_batching(false),
     ov::enable_profiling(false)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::num_requests(2),
     ov::hint::allow_auto_batching(true),
     ov::enable_profiling(false)}};
INSTANTIATE_TEST_SUITE_P(smoke_AutoCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_setcore_properties),
                                            ::testing::ValuesIn(auto_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
        {ov::enable_profiling(false)},
        {ov::log::level("LOG_NONE")},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::allow_auto_batching(true)},
        {ov::auto_batch_timeout("1000")},
        {ov::hint::execution_mode(ov::hint::ExecutionMode::UNDEFINED)},
        {ov::intel_auto::device_bind_buffer(false)},
        {ov::intel_auto::enable_startup_fallback(true)},
        {ov::device::priorities("")}
};
INSTANTIATE_TEST_SUITE_P(smoke_AutoBehaviorTests, OVPropertiesDefaultTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(default_properties)),
        OVPropertiesDefaultTests::getTestCaseName);

const std::vector<std::pair<ov::AnyMap, std::string>> automultiExeDeviceConfigs = {
            std::make_pair(ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_CPU)}}, "CPU")
    };

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiCompileModelBehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(automultiExeDeviceConfigs)),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_multi_device_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU", ov::num_streams(4), ov::enable_profiling(true))}};

const std::vector<ov::AnyMap> auto_multi_incorrect_device_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::num_streams(4),
     ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::num_streams(4),
     ov::device::properties("CPU", ov::num_streams(4), ov::enable_profiling(true))}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsNoThrow,
                         OVSetSupportPropComplieModleWithoutConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_multi_device_properties)),
                         OVSetSupportPropComplieModleWithoutConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsThrow,
                         OVSetUnsupportPropComplieModleWithoutConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_multi_incorrect_device_properties)),
                         OVSetUnsupportPropComplieModleWithoutConfigTests::getTestCaseName);
} // namespace
