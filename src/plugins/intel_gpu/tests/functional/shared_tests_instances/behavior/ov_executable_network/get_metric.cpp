// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU"));


const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_GPU = {
    {"GPU", std::make_pair(ov::AnyMap{}, "GPU.0")},
    {"GPU.0", std::make_pair(ov::AnyMap{}, "GPU.0")},
    {"BATCH:GPU", std::make_pair(ov::AnyMap{}, "GPU.0")}};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::ValuesIn(GetMetricTest_ExecutionDevice_GPU));

auto multiDevicePriorityConfigs = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(ov::test::utils::DEVICE_CPU)},
                                   {ov::device::priorities(ov::test::utils::DEVICE_GPU)},
                                   {ov::device::priorities(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU)}};
};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs())));

auto multiModelPriorityConfigs = []() {
    return std::vector<ov::AnyMap>{{ov::hint::model_priority(ov::hint::Priority::HIGH)},
                                   {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
                                   {ov::hint::model_priority(ov::hint::Priority::LOW)}};
};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs())),
                         OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY::getTestCaseName);

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetConfigTest,
                         OVClassCompiledModelGetConfigTest,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelSetIncorrectConfigTest,
                         OVClassCompiledModelSetIncorrectConfigTest,
                         ::testing::Values("GPU"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(smoke_OVCompiledModelIncorrectDevice, OVCompiledModelIncorrectDevice, ::testing::Values("GPU"));

const std::vector<ov::AnyMap> incorrect_device_priorities_properties = {{ov::device::priorities("NONE")},
                                                                        {ov::device::priorities("NONE", "GPU")},
                                                                        {ov::device::priorities("-", "GPU")},
                                                                        {ov::device::priorities("-NONE", "CPU")},
                                                                        {ov::device::priorities("-CPU", "-NONE")},
                                                                        {ov::device::priorities("-NONE", "-NONE")}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorIncorrectPropertiesTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI,
                                                              ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(incorrect_device_priorities_properties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> gpuCorrectConfigs = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::hint::allow_auto_batching(false)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::hint::allow_auto_batching(true)}};

auto gpuCorrectConfigsWithSecondaryProperties = []() {
    return std::vector<ov::AnyMap>{
        {ov::device::properties(ov::test::utils::DEVICE_GPU,
                                ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE),
                                ov::hint::inference_precision(ov::element::f32))},
        {ov::device::properties(ov::test::utils::DEVICE_GPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::allow_auto_batching(false))},
        {ov::device::properties(ov::test::utils::DEVICE_GPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::allow_auto_batching(false)),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                ov::hint::allow_auto_batching(false))}};
};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompileModelWithCorrectPropertiesAutoBatchingTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(gpuCorrectConfigs)));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompileModelWithCorrectSecondaryPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(gpuCorrectConfigsWithSecondaryProperties())),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_AUTO_OVClassCompileModelWithCorrectSecondaryPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO:GPU", "MULTI:GPU", "HETERO:GPU"),
                                            ::testing::ValuesIn(gpuCorrectConfigsWithSecondaryProperties())));

auto autoCorrectConfigs = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(ov::test::utils::DEVICE_GPU),
                                    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                    ov::hint::allow_auto_batching(false)},
                                   {ov::device::priorities(ov::test::utils::DEVICE_GPU),
                                    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                    ov::hint::allow_auto_batching(true)}};
};

auto autoCorrectConfigsWithSecondaryProperties = []() {
    return std::vector<ov::AnyMap>{
        {ov::device::priorities(ov::test::utils::DEVICE_GPU),
         ov::device::properties(ov::test::utils::DEVICE_AUTO,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::allow_auto_batching(false))},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU),
         ov::device::properties(ov::test::utils::DEVICE_GPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::allow_auto_batching(false))},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU),
         ov::device::properties(ov::test::utils::DEVICE_GPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::allow_auto_batching(false)),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                ov::hint::allow_auto_batching(false))},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU),
         ov::device::properties("GPU.0",
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::allow_auto_batching(false)),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                ov::hint::allow_auto_batching(false))}};
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_OVClassCompileModelWithCorrectPropertiesAutoBatchingTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI,
                                                              ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoCorrectConfigs())));

INSTANTIATE_TEST_SUITE_P(smoke_Auto_OVClassCompileModelWithCorrectSecondaryPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI,
                                                              ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoCorrectConfigsWithSecondaryProperties())),
                         ::testing::PrintToStringParamName());

const std::vector<ov::AnyMap> batchCorrectConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_Batch_OVClassCompileModelWithCorrectPropertiesAutoBatchingTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("BATCH:GPU"), ::testing::ValuesIn(batchCorrectConfigs)));

const std::vector<std::pair<ov::AnyMap, std::string>> autoExeDeviceConfigs = {
    std::make_pair(ov::AnyMap{{ov::device::priorities("GPU.0")}}, "GPU.0"),
#ifdef ENABLE_INTEL_CPU
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU)}},
                   "undefined"),
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU)}},
                   "CPU"),
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU),
                               ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                   "CPU,GPU"),
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
                               ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                   "GPU,CPU"),
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
                               ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                               ov::hint::allow_auto_batching(true)}},
                   "GPU,CPU"),
#endif
};

const std::vector<std::pair<ov::AnyMap, std::string>> multiExeDeviceConfigs = {
    std::make_pair(ov::AnyMap{{ov::device::priorities("GPU.0")}}, "GPU.0"),
#ifdef ENABLE_INTEL_CPU
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU)}},
                   "GPU,CPU"),
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU)}},
                   "CPU,GPU"),
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU),
                               ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                   "CPU,GPU"),
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
                               ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                   "GPU,CPU"),
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
                               ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                               ov::hint::allow_auto_batching(true)}},
                   "GPU,CPU"),
#endif
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiCompileModelBehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoExeDeviceConfigs)),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultiCompileModelBehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiExeDeviceConfigs)),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

}  // namespace
