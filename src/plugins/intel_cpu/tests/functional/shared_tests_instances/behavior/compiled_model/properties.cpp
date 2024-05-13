// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::num_streams(-100)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
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

const std::vector<ov::AnyMap> default_properties = {
    {ov::affinity(default_affinity)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(default_properties)),
                         OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         OVCompiledModelPropertiesDefaultSupportedTests::getTestCaseName);

const std::vector<ov::AnyMap> properties = {{ov::num_streams(ov::streams::NUMA)},
                                            {ov::num_streams(ov::streams::AUTO)},
                                            {ov::num_streams(0), ov::inference_num_threads(1)},
                                            {ov::num_streams(1), ov::inference_num_threads(1)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVCompiledModelIncorrectDevice, OVCompiledModelIncorrectDevice, ::testing::Values("CPU"));

const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
    {ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

// OV Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_CPUOVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

//
// OV CompiledModel Get RO Property
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("CPU"));

const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_CPU = {
        {"CPU", std::make_pair(ov::AnyMap{}, "CPU")}};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelGetPropertyTest, OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
        ::testing::ValuesIn(GetMetricTest_ExecutionDevice_CPU));

//
// OV CompiledModel GetProperty / SetProperty
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelGetConfigTest, OVClassCompiledModelGetConfigTest,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelSetIncorrectConfigTest, OVClassCompiledModelSetIncorrectConfigTest,
        ::testing::Values("CPU"));
}  // namespace
