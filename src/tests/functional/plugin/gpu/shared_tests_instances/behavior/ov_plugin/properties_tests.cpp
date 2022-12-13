// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include <openvino/runtime/auto/properties.hpp>

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {

const std::vector<ov::AnyMap> gpu_properties = {
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                ::testing::ValuesIn(gpu_properties)),
        OVPropertiesTests::getTestCaseName);

auto auto_multi_properties = []() {
    return std::vector<ov::AnyMap>{
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::intel_auto::device_bind_buffer("YES")},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::intel_auto::device_bind_buffer("NO")}};
};

const std::vector<ov::AnyMap> multi_properties = {
        {ov::device::priorities("CPU", "GPU")},
        {ov::device::priorities("CPU(1)", "GPU")},
        {ov::device::priorities("CPU(1)", "GPU(2)")}
};

const std::vector<ov::AnyMap> auto_properties = {
        {ov::device::priorities("CPU", "GPU")},
        {ov::device::priorities("-CPU", "GPU")},
        {ov::device::priorities("CPU(1)", "GPU")},
        {ov::device::priorities("CPU(1)", "GPU(2)")},
        {ov::device::priorities("CPU", "-GPU")}
};


const std::vector<ov::AnyMap> auto_Multi_compiled_empty_properties = {
        {}
};

const std::vector<ov::AnyMap> multi_plugin_Incorrect_properties = {
        {ov::device::priorities("NONE")}
};
const std::vector<ov::AnyMap> auto_plugin_Incorrect_properties = {
        {ov::device::priorities("NONE", "GPU")},
        {ov::device::priorities("-", "GPU")},
        {ov::device::priorities("-NONE", "CPU")},
        {ov::device::priorities("-CPU", "-NONE")},
        {ov::device::priorities("-NONE", "-NONE")}
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiBehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO, CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(auto_multi_properties())),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(auto_properties)),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultiBehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(multi_properties)),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBehaviorIncorrectPropertiesTests, OVSetPropComplieModleWihtIncorrectPropTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(auto_plugin_Incorrect_properties),
                ::testing::ValuesIn(auto_Multi_compiled_empty_properties)),
        OVSetPropComplieModleWihtIncorrectPropTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultiBehaviorIncorrectPropertiesTests, OVSetPropComplieModleWihtIncorrectPropTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(multi_plugin_Incorrect_properties),
                ::testing::ValuesIn(auto_Multi_compiled_empty_properties)),
        OVSetPropComplieModleWihtIncorrectPropTests::getTestCaseName);

const std::vector<ov::AnyMap> gpu_plugin_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::num_requests(2),
     ov::enable_profiling(false)}};
const std::vector<ov::AnyMap> gpu_compileModel_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::num_requests(10),
     ov::enable_profiling(true)}};

INSTANTIATE_TEST_SUITE_P(smoke_gpuCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_GPU),
                                            ::testing::ValuesIn(gpu_plugin_properties),
                                            ::testing::ValuesIn(gpu_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

auto auto_multi_plugin_properties = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(CommonTestUtils::DEVICE_GPU),
                                    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                    ov::hint::num_requests(2),
                                    ov::hint::allow_auto_batching(false),
                                    ov::enable_profiling(false)}};
};
auto auto_multi_compileModel_properties = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(CommonTestUtils::DEVICE_GPU),
                                    ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                    ov::hint::num_requests(10),
                                    ov::hint::allow_auto_batching(true),
                                    ov::enable_profiling(true)}};
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_multi_plugin_properties()),
                                            ::testing::ValuesIn(auto_multi_compileModel_properties())),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<std::pair<ov::AnyMap, std::string>> autoExeDeviceConfigs = {
            std::make_pair(ov::AnyMap{{ov::device::priorities("GPU.0")}}, "GPU.0"),
            #ifdef ENABLE_INTEL_CPU
            std::make_pair(ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_GPU, CommonTestUtils::DEVICE_CPU)}}, "undefined"),
            std::make_pair(ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_GPU)}}, "CPU"),
            std::make_pair(ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_GPU),
                                        ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}}, "CPU,GPU"),
            std::make_pair(ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_GPU, CommonTestUtils::DEVICE_CPU),
                                        ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}}, "GPU,CPU"),
            #endif
    };

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiCompileModelBehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoExeDeviceConfigs)),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

} // namespace
