// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;

namespace {

std::vector<ov::AnyMap> operator+(std::vector<ov::AnyMap> origAnyMapVector,
                                  const std::vector<std::pair<std::string, ov::Any>>& newPair) {
    std::vector<ov::AnyMap> newAnyMapVector(origAnyMapVector.size() * newPair.size());
    size_t index = 0;
    for (const auto& pair : newPair) {
        for (auto&& anyMap : origAnyMapVector) {
            ov::AnyMap newAnyMap = anyMap;
            ov::AnyMap& newAnyMapRef = (anyMap.find(ov::device::properties.name()) != anyMap.end())
                                           ? newAnyMap.find(ov::device::properties.name())
                                                 ->second.as<ov::AnyMap>()
                                                 .begin()
                                                 ->second.as<ov::AnyMap>()
                                           : newAnyMap;

            if (newAnyMapRef.find(pair.first) == newAnyMapRef.end()) {
                newAnyMapRef.emplace(pair);
                newAnyMapVector.at(index) = newAnyMap;
                ++index;
            }
        }
    }
    newAnyMapVector.resize(index);
    return newAnyMapVector;
}

const std::vector<std::pair<std::string, ov::Any>> compiledModelProperties = {
    {ov::enable_profiling.name(), ov::Any(true)},
    {ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::num_requests.name(), ov::Any(4u)},
    {ov::hint::inference_precision.name(), ov::Any(ov::element::i8)},
    {ov::hint::enable_cpu_pinning.name(), ov::Any(true)},
    {ov::hint::model_priority.name(), ov::Any(ov::hint::Priority::HIGH)},
    {ov::intel_npu::tiles.name(), ov::Any(2)},
    {ov::intel_npu::profiling_type.name(), ov::Any(ov::intel_npu::ProfilingType::INFER)},
    {ov::intel_npu::defer_weights_load.name(), ov::Any(false)}};

const std::string& expectedModelName = []() -> std::string {
    return ov::test::behavior::getDefaultNGraphFunctionForTheDevice()->get_friendly_name();
}();

const std::vector<ov::AnyMap> publicCompiledModelConfigs = {
    {{ov::device::id.name(), ov::Any("")}},
    {{ov::hint::enable_cpu_pinning.name(), ov::Any(false)}},
    {{ov::hint::model_priority.name(), ov::Any(ov::hint::Priority::MEDIUM)}},
    {{ov::execution_devices.name(), ov::Any(ov::test::utils::DEVICE_NPU)}},
    {{ov::hint::execution_mode.name(), ov::Any(ov::hint::ExecutionMode::PERFORMANCE)}},
    {{ov::hint::inference_precision.name(), ov::Any(ov::element::f16)}},
    {{ov::loaded_from_cache.name(), ov::Any(false)}},
    {{ov::model_name.name(), ov::Any(expectedModelName)}},
    {{ov::optimal_number_of_infer_requests.name(), ov::Any(1u)}},
    {{ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::LATENCY)}},
    {{ov::hint::num_requests.name(), ov::Any(1u)}},
    {{ov::enable_profiling.name(), ov::Any(false)}}};

const std::vector<ov::AnyMap> compiledModelIncorrectConfigs = {
    {{"NPU_INEXISTENT_PROPERTY", "NPU_INEXISTENT_PROPERTY_VALUE"}}};

const std::vector<std::pair<std::string, ov::Any>> allModelPriorities = {
    ov::hint::model_priority(ov::hint::Priority::LOW),
    ov::hint::model_priority(ov::hint::Priority::MEDIUM),
    ov::hint::model_priority(ov::hint::Priority::HIGH)};

const std::vector<std::pair<std::string, std::string>>& compiledModelPropertiesAnyToString =
    []() -> std::vector<std::pair<std::string, std::string>> {
    std::vector<std::pair<std::string, std::string>> compiledModelProps(compiledModelProperties.size());
    for (auto it = compiledModelProperties.cbegin(); it != compiledModelProperties.cend(); ++it) {
        auto&& distance = it - compiledModelProperties.cbegin();
        compiledModelProps.at(distance) = {it->first, it->second.as<std::string>()};
    }
    return compiledModelProps;
}();

const std::vector<ov::AnyMap>& compiledModelConfigs = []() -> std::vector<ov::AnyMap> {
    std::vector<ov::AnyMap> compiledModelConfigsMap(compiledModelProperties.size());
    for (auto it = compiledModelProperties.cbegin(); it != compiledModelProperties.cend(); ++it) {
        auto&& distance = it - compiledModelProperties.cbegin();
        compiledModelConfigsMap.at(distance) = {*it};
    }
    return compiledModelConfigsMap;
}();

const auto& heteroCompiledModelConfigs = []() -> std::vector<ov::AnyMap> {
    std::vector<ov::AnyMap> heteroConfigs(compiledModelConfigs.size());
    for (auto it = compiledModelConfigs.cbegin(); it != compiledModelConfigs.cend(); ++it) {
        auto&& distance = it - compiledModelConfigs.cbegin();
        heteroConfigs.at(distance) = {
            ov::device::priorities(ov::test::utils::DEVICE_NPU),
            {ov::device::properties.name(), ov::Any(ov::AnyMap{{ov::test::utils::DEVICE_NPU, ov::Any(*it)}})}};
    }
    return heteroConfigs;
}();

const auto& combineParamsExecDevices = []() -> std::vector<std::pair<ov::AnyMap, std::string>> {
    std::vector<std::pair<ov::AnyMap, std::string>> execParams(compiledModelConfigs.size());
    for (auto it = compiledModelConfigs.cbegin(); it != compiledModelConfigs.cend(); ++it) {
        auto&& distance = it - compiledModelConfigs.cbegin();
        execParams.at(distance) = std::make_pair(*it, ov::test::utils::DEVICE_NPU);
    }
    return execParams;
}();

const auto& combineHeteroParamsExecDevices = []() -> std::vector<std::pair<ov::AnyMap, std::string>> {
    std::vector<std::pair<ov::AnyMap, std::string>> execHeteroParams(heteroCompiledModelConfigs.size());
    for (auto it = heteroCompiledModelConfigs.cbegin(); it != heteroCompiledModelConfigs.cend(); ++it) {
        auto&& distance = it - heteroCompiledModelConfigs.cbegin();
        execHeteroParams.at(distance) = std::make_pair(*it, ov::test::utils::DEVICE_NPU);
    }
    return execHeteroParams;
}();

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetIncorrectPropertyTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                           ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetIncorrectPropertyTest>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetPropertyTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                           ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetPropertyTest>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVClassCompiledModelGetConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetConfigTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelPropertiesTests>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(heteroCompiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelPropertiesTests>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompileModelWithCorrectPropertiesTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(heteroCompiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompileModelWithCorrectPropertiesTest>);

const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
    {ov::device::properties(ov::test::utils::DEVICE_NPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::properties(ov::test::utils::DEVICE_NPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties(ov::test::utils::DEVICE_NPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> multiConfigsWithSecondaryProperties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties(ov::test::utils::DEVICE_CPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties(ov::test::utils::DEVICE_CPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties(ov::test::utils::DEVICE_NPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> autoConfigsWithSecondaryProperties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties(ov::test::utils::DEVICE_CPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties(ov::test::utils::DEVICE_CPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties(ov::test::utils::DEVICE_NPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties(ov::test::utils::DEVICE_CPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::device::priorities(ov::test::utils::DEVICE_NPU),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties(ov::test::utils::DEVICE_CPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties(ov::test::utils::DEVICE_NPU,
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_OVClassLoadNetworkWithCorrectSecondaryPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU, "AUTO:NPU", "MULTI:NPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Multi_BehaviorTests_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(multiConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_AUTO_BehaviorTests_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelIncorrectConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelPropertiesIncorrectTests>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelIncorrectDevice,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledModelIncorrectDevice>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVCompiledModelPropertiesDefaultSupportedTests>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(publicCompiledModelConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelPropertiesDefaultTests>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelSetCorrectConfigTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelPropertiesAnyToString)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelSetCorrectConfigTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelSetCorrectConfigTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compiledModelPropertiesAnyToString)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelSetCorrectConfigTest>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVClassCompiledModelSetIncorrectConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelSetIncorrectConfigTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelSetIncorrectConfigTest,
                         ::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                           ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelSetIncorrectConfigTest>);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(compiledModelConfigs + allModelPriorities)),
    ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY>);

INSTANTIATE_TEST_SUITE_P(
    smoke_Hetero_BehaviorTests,
    OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY,
    ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                         ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(heteroCompiledModelConfigs + allModelPriorities)),
    ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY>);

INSTANTIATE_TEST_SUITE_P(
    smoke_Hetero_BehaviorTests,
    OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY,
    ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                         ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(heteroCompiledModelConfigs)),
    ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(combineParamsExecDevices)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetPropertyTest_EXEC_DEVICES>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(combineHeteroParamsExecDevices)),
                         ov::test::utils::appendPlatformTypeTestName<OVClassCompiledModelGetPropertyTest_EXEC_DEVICES>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(combineParamsExecDevices)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileModelGetExecutionDeviceTests>);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_HETERO) + ":" +
                                                              ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(combineHeteroParamsExecDevices)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileModelGetExecutionDeviceTests>);

}  // namespace
