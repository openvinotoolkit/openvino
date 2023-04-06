// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
#include <openvino/runtime/properties.hpp>
#include "ie_system_conf.h"
#include "openvino/runtime/core.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCommon, OVClassBasicTestP,
        ::testing::Values(std::make_pair("openvino_intel_cpu_plugin", "CPU")));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassNetworkTestP, OVClassNetworkTestP,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, OVClassImportExportTestP,
        ::testing::Values("HETERO:CPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest,
                         OVClassSetEnableHyperThreadingHintConfigTest,
                         ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest,
                         OVClassSetSchedulingCoreTypeHintConfigTest,
                         ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetAvailableDevices, OVClassGetAvailableDevices,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest, OVClassSetEnableCpuPinningHintConfigTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassSetModelPriorityConfigTest, OVClassSetModelPriorityConfigTest,
        ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassSetTBBForceTerminatePropertyTest, OVClassSetTBBForceTerminatePropertyTest,
        ::testing::Values("AUTO", "GPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassSetLogLevelConfigTest, OVClassSetLogLevelConfigTest,
        ::testing::Values("MULTI", "AUTO"));

const std::vector<ov::AnyMap> multiConfigs = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}
};
const std::vector<ov::AnyMap> configsDeviceProperties = {{ov::device::properties("CPU", ov::num_streams(3))},
                                                         {ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(3)}}})}};
const std::vector<ov::AnyMap> configsDevicePropertiesDouble = {{ov::device::properties("CPU", ov::num_streams(5)),
                                                                ov::num_streams(3)},
                                                               {ov::device::properties("CPU", ov::num_streams(5)),
                                                                ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(7)}}}),
                                                                ov::num_streams(3)},
                                                               {ov::device::properties("CPU", ov::num_streams(3)),
                                                                ov::device::properties("CPU", ov::num_streams(5))},
                                                               {ov::device::properties("CPU", ov::num_streams(3)),
                                                                ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(5)}}})},
                                                               {ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(3)}}}),
                                                                ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(5)}}})}};
const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
    {ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> multiConfigsWithSecondaryProperties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> autoConfigsWithSecondaryProperties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_GPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::device::priorities(CommonTestUtils::DEVICE_CPU),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> heteroConfigsWithSecondaryProperties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("HETERO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("HETERO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(CommonTestUtils::DEVICE_GPU),
     ov::device::properties("HETERO",
                            ov::enable_profiling(false),
                            ov::device::priorities(CommonTestUtils::DEVICE_CPU),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassSetDevicePriorityConfigTest, OVClassSetDevicePriorityConfigTest,
        ::testing::Combine(::testing::Values("MULTI", "AUTO", "HETERO"),
                           ::testing::ValuesIn(multiConfigs)));
//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest,
        ::testing::Values("CPU"));

//////////////////////////////////////////////////////////////////////////////////////////

TEST(OVClassBasicTest, smoke_SetConfigInferenceNumThreads) {
    ov::Core ie;
    int32_t value = 0;
    int32_t num_threads = 1;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::inference_num_threads(num_threads)));
    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::inference_num_threads));
    ASSERT_EQ(num_threads, value);

    num_threads = 4;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::inference_num_threads(num_threads)));
    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::inference_num_threads));
    ASSERT_EQ(num_threads, value);
}

TEST(OVClassBasicTest, smoke_SetConfigStreamsNum) {
    ov::Core ie;
    int32_t value = 0;
    int32_t num_streams = 1;

    auto setGetProperty = [&ie](int32_t& getProperty, int32_t setProperty){
        OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::num_streams(setProperty)));
        OV_ASSERT_NO_THROW(getProperty = ie.get_property("CPU", ov::num_streams));
    };

    setGetProperty(value, num_streams);
    ASSERT_EQ(num_streams, value);

    num_streams = ov::streams::NUMA;

    setGetProperty(value, num_streams);
    ASSERT_GT(value, 0); // value has been configured automatically

    num_streams = ov::streams::AUTO;

    setGetProperty(value, num_streams);
    ASSERT_GT(value, 0); // value has been configured automatically
}

TEST(OVClassBasicTest, smoke_SetConfigAffinity) {
    ov::Core ie;
    ov::Affinity value = ov::Affinity::NONE;

#if (defined(__APPLE__) || defined(_WIN32))
    auto numaNodes = InferenceEngine::getAvailableNUMANodes();
    auto coreTypes = InferenceEngine::getAvailableCoresTypes();
    auto defaultBindThreadParameter = ov::Affinity::NONE;
    if (coreTypes.size() > 1) {
        defaultBindThreadParameter = ov::Affinity::HYBRID_AWARE;
    } else if (numaNodes.size() > 1) {
        defaultBindThreadParameter = ov::Affinity::NUMA;
    }
#else
    auto defaultBindThreadParameter = ov::Affinity::CORE;
    auto coreTypes = InferenceEngine::getAvailableCoresTypes();
    if (coreTypes.size() > 1) {
        defaultBindThreadParameter = ov::Affinity::HYBRID_AWARE;
    }
#endif
    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::affinity));
    ASSERT_EQ(defaultBindThreadParameter, value);

    const ov::Affinity affinity = defaultBindThreadParameter == ov::Affinity::HYBRID_AWARE ? ov::Affinity::NUMA : ov::Affinity::HYBRID_AWARE;
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::affinity(affinity)));
    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::affinity));
    ASSERT_EQ(affinity, value);
}

TEST(OVClassBasicTest, smoke_SetConfigHintInferencePrecision) {
    ov::Core ie;
    auto value = ov::element::f32;
    const auto precision = InferenceEngine::with_cpu_x86_bfloat16() ? ov::element::bf16 : ov::element::f32;

    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(precision, value);

    const auto forcedPrecision = ov::element::f32;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(forcedPrecision)));
    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(value, forcedPrecision);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto forced_precision_deprecated = ov::element::f32;
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(forced_precision_deprecated)));
    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(value, forced_precision_deprecated);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST(OVClassBasicTest, smoke_SetConfigEnableProfiling) {
    ov::Core ie;
    auto value = false;
    const bool enableProfilingDefault = false;

    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::enable_profiling));
    ASSERT_EQ(enableProfilingDefault, value);

    const bool enableProfiling = true;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::enable_profiling(enableProfiling)));
    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::enable_profiling));
    ASSERT_EQ(enableProfiling, value);
}

const auto bf16_if_supported       = InferenceEngine::with_cpu_x86_bfloat16() ?    ov::element::bf16 : ov::element::f32;
const auto bf16_if_can_be_emulated = InferenceEngine::with_cpu_x86_avx512_core() ? ov::element::bf16 : ov::element::f32;
using ExpectedModeAndType = std::pair<ov::hint::ExecutionMode, ov::element::Type>;

const std::map<ov::hint::ExecutionMode, ExpectedModeAndType> exectedTypeByMode {
    {ov::hint::ExecutionMode::PERFORMANCE, {ov::hint::ExecutionMode::PERFORMANCE,
                                            bf16_if_supported}},
    {ov::hint::ExecutionMode::ACCURACY,    {ov::hint::ExecutionMode::ACCURACY,
                                            ov::element::f32}},
};

TEST(OVClassBasicTest, smoke_SetConfigExecutionModeExpectCorrespondingInferencePrecision) {
    ov::Core ie;
    const auto inference_precision_default = bf16_if_supported;
    const auto execution_mode_default = ov::hint::ExecutionMode::PERFORMANCE;
    auto execution_mode_value = ov::hint::ExecutionMode::PERFORMANCE;
    auto inference_precision_value = ov::element::undefined;

    // check default values
    OV_ASSERT_NO_THROW(inference_precision_value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, inference_precision_default);
    OV_ASSERT_NO_THROW(execution_mode_value = ie.get_property("CPU", ov::hint::execution_mode));
    ASSERT_EQ(execution_mode_value, execution_mode_default);

    for (const auto& m : exectedTypeByMode) {
        const auto execution_mode = m.first;
        const auto execution_mode_exected = m.second.first;
        const auto inference_precision_exected = m.second.second;

        OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(execution_mode)));
        OV_ASSERT_NO_THROW(execution_mode_value = ie.get_property("CPU", ov::hint::execution_mode));
        ASSERT_EQ(execution_mode_value, execution_mode_exected);

        OV_ASSERT_NO_THROW(inference_precision_value = ie.get_property("CPU", ov::hint::inference_precision));
        ASSERT_EQ(inference_precision_value, inference_precision_exected);
    }
}

TEST(OVClassBasicTest, smoke_SetConfigExecutionModeAndInferencePrecision) {
    ov::Core ie;
    const auto inference_precision_default = bf16_if_supported;
    const auto execution_mode_default = ov::hint::ExecutionMode::PERFORMANCE;

    auto expect_execution_mode = [&](const ov::hint::ExecutionMode expected_value) {
        auto execution_mode_value = ov::hint::ExecutionMode::ACCURACY;
        OV_ASSERT_NO_THROW(execution_mode_value = ie.get_property("CPU", ov::hint::execution_mode));
        ASSERT_EQ(execution_mode_value, expected_value);
    };

    auto expect_inference_precision = [&](const ov::element::Type expected_value) {
        auto inference_precision_value = ov::element::undefined;;
        OV_ASSERT_NO_THROW(inference_precision_value = ie.get_property("CPU", ov::hint::inference_precision));
        ASSERT_EQ(inference_precision_value, expected_value);
    };

    // check default values
    expect_execution_mode(execution_mode_default);
    expect_inference_precision(inference_precision_default);
    // verify that conflicting property values work as expect
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)));
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));
    expect_execution_mode(ov::hint::ExecutionMode::PERFORMANCE); // inference_preicision does not affect execution_mode property itself
    expect_inference_precision(ov::element::f32); // inference_preicision has more priority than performance mode

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)));
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(bf16_if_can_be_emulated)));
    expect_execution_mode(ov::hint::ExecutionMode::ACCURACY);
    expect_inference_precision(bf16_if_can_be_emulated);
}

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest,
        ::testing::Values("CPU"));

// IE Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassLoadNetworkWithCorrectSecondaryPropertiesTest,
                         OVClassLoadNetworkWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU", "AUTO:CPU", "MULTI:CPU", "HETERO:CPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_Multi_OVClassLoadNetworkWithSecondaryPropertiesTest,
                         OVClassLoadNetworkWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(multiConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_AUTO_OVClassLoadNetworkWithSecondaryPropertiesTest,
                         OVClassLoadNetworkWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_HETERO_OVClassLoadNetworkWithSecondaryPropertiesTest,
                         OVClassLoadNetworkWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("HETERO"),
                                            ::testing::ValuesIn(heteroConfigsWithSecondaryProperties)));

// IE Class load and check network with ov::device::properties
INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassLoadNetworkAndCheckWithSecondaryPropertiesTest,
                         OVClassLoadNetworkAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsDeviceProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassLoadNetworkAndCheckWithSecondaryPropertiesDoubleTest,
                         OVClassLoadNetworkAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsDevicePropertiesDouble)));
INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest,
        ::testing::Values("CPU"));

const std::vector<ov::AnyMap> auto_multi_default_properties = {{}, {ov::hint::allow_auto_batching(true)}};
INSTANTIATE_TEST_SUITE_P(smoke_AUTO_MULTI_ReturnDefaultHintTest,
                         OVClassLoadNetWorkReturnDefaultHintTest,
                         ::testing::Combine(::testing::Values("AUTO:CPU", "MULTI:CPU"),
                                            ::testing::ValuesIn(auto_multi_default_properties)));
// For AUTO, User sets perf_hint, AUTO's perf_hint should not return default value LATENCY
const std::vector<ov::AnyMap> default_auto_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}};
// For MULTI, User sets perf_hint or Affinity or num_streams or infer_num_threads, MULTI's perf_hint should
// not return default value THROUGHPUT
// For Secondary property test about default hint is in auto_load_network_properties_test.cpp
const std::vector<ov::AnyMap> default_multi_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::affinity(ov::Affinity::NONE)},
    {ov::num_streams(ov::streams::AUTO)},
    {ov::inference_num_threads(1)}};
INSTANTIATE_TEST_SUITE_P(smoke_AUTO_DoNotReturnDefaultHintTest,
                         OVClassLoadNetWorkDoNotReturnDefaultHintTest,
                         ::testing::Combine(::testing::Values("AUTO:CPU"),
                                            ::testing::ValuesIn(default_auto_properties)));
INSTANTIATE_TEST_SUITE_P(smoke_MULTI_DoNotReturnDefaultHintTest,
                         OVClassLoadNetWorkDoNotReturnDefaultHintTest,
                         ::testing::Combine(::testing::Values("MULTI:CPU"),
                                            ::testing::ValuesIn(default_multi_properties)));

const std::vector<ov::AnyMap> configsWithEmpty = {{}};
const std::vector<ov::AnyMap> configsWithMetaPlugin = {{ov::device::priorities("AUTO")},
                                                       {ov::device::priorities("MULTI")},
                                                       {ov::device::priorities("AUTO", "MULTI")},
                                                       {ov::device::priorities("AUTO", "CPU")},
                                                       {ov::device::priorities("MULTI", "CPU")}};

INSTANTIATE_TEST_SUITE_P(
    smoke_MULTI_AUTO_DoNotSupportMetaPluginLoadingItselfRepeatedlyWithEmptyConfigTest,
    OVClassLoadNetworkWithCondidateDeviceListContainedMetaPluginTest,
    ::testing::Combine(::testing::Values("MULTI:AUTO", "AUTO:MULTI", "MULTI:CPU,AUTO", "AUTO:CPU,MULTI"),
                       ::testing::ValuesIn(configsWithEmpty)),
    ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_MULTI_AUTO_DoNotSupportMetaPluginLoadingItselfRepeatedlyTest,
                         OVClassLoadNetworkWithCondidateDeviceListContainedMetaPluginTest,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(configsWithMetaPlugin)),
                         ::testing::PrintToStringParamName());
}  // namespace
