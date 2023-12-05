// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"
#include "ie_system_conf.h"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::num_streams(-100)},
};

const std::vector<ov::AnyMap> auto_batch_inproperties = {
    {ov::num_streams(-100)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU,
                                                              ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

#if (defined(__APPLE__) || defined(_WIN32))
auto default_affinity = [] {
    auto numaNodes = InferenceEngine::getAvailableNUMANodes();
    auto coreTypes = InferenceEngine::getAvailableCoresTypes();
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
    auto coreTypes = InferenceEngine::getAvailableCoresTypes();
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
                                            {ov::num_streams(1), ov::inference_num_threads(1)},
                                            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                                              InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}};

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU), ov::num_streams(ov::streams::AUTO)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
      InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
};

const std::vector<ov::AnyMap> auto_batch_properties = {
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_CPU) + "(4)"}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_CPU) + "(4)"},
     {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "1"}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_CPU) + "(4)"},
     {ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
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

const std::vector<ov::AnyMap> heteroConfigsWithSecondaryProperties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("HETERO",
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
     ov::device::properties("HETERO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_GPU),
     ov::device::properties("HETERO",
                            ov::enable_profiling(false),
                            ov::device::priorities(ov::test::utils::DEVICE_CPU),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

// IE Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_CPUOVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU", "HETERO:CPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_HETERO_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("HETERO"),
                                            ::testing::ValuesIn(heteroConfigsWithSecondaryProperties)));
}  // namespace
