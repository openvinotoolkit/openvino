// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    OVClassCompiledModelPropertiesIncorrectTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI, "AUTO:TEMPLATE", ov::test::utils::DEVICE_AUTO),
                       ::testing::ValuesIn(inproperties)),
    OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

#if (defined(__APPLE__) || defined(_WIN32))
auto default_affinity = [] {
    auto numaNodes = ov::get_available_numa_nodes();
    auto coreTypes = ov::get_available_cores_types();
    if (coreTypes.size() > 1) {
        return ov::Affinity::HYBRID_AWARE;
    } else if (numaNodes.size() > 1) {
        return ov::Affinity::NUMA;
    } else {
        return ov::Affinity::NONE;
    }
}();
#else
auto default_affinity = [] {
    auto coreTypes = ov::get_available_cores_types();
    if (coreTypes.size() > 1) {
        return ov::Affinity::HYBRID_AWARE;
    } else {
        return ov::Affinity::CORE;
    }
}();
#endif

const std::vector<ov::AnyMap> multi_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::num_streams(ov::streams::AUTO)},
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVCompiledModelIncorrectDevice,
                         OVCompiledModelIncorrectDevice,
                         ::testing::Values("TEMPLATE"));

const std::vector<ov::AnyMap> auto_multi_device_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::device::properties("TEMPLATE", ov::num_streams(4))},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::device::properties("TEMPLATE", ov::num_streams(4), ov::enable_profiling(true))},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::device::properties(ov::AnyMap{{"TEMPLATE", ov::AnyMap{{ov::num_streams(4), ov::enable_profiling(true)}}}})}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsNoThrow,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_multi_device_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
    {ov::device::properties("TEMPLATE", ov::num_streams(4))},
    {ov::device::properties("TEMPLATE",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))}};

const std::vector<ov::AnyMap> autoConfigsWithSecondaryProperties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::device::properties("TEMPLATE",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("TEMPLATE",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))}};

// OV Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_CPUOVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO:TEMPLATE", "MULTI:TEMPLATE"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_Multi_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_AUTO_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

const std::vector<std::pair<ov::AnyMap, std::string>> automultiExeDeviceConfigs = {
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}}, "TEMPLATE")};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiCompileModelBehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(automultiExeDeviceConfigs)),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)),
                         OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY::getTestCaseName);

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {{ov::hint::model_priority(ov::hint::Priority::HIGH)},
                                                           {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
                                                           {ov::hint::model_priority(ov::hint::Priority::LOW)},
                                                           {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO:TEMPLATE"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));

const std::vector<ov::AnyMap> auto_default_properties = {
    {ov::enable_profiling(false)},
    {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_Default_test,
                         OVClassCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_default_properties)),
                         OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

const std::vector<ov::AnyMap> multi_default_properties = {{ov::enable_profiling(false)}};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_Default_test,
                         OVClassCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(multi_default_properties)),
                         OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

}  // namespace
