// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include <openvino/runtime/auto/properties.hpp>

#include "ie_system_conf.h"

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCommon,
                         OVClassBasicPropsTestP,
                         ::testing::Values(std::make_pair("openvino_intel_cpu_plugin", "CPU")));

//////////////////////////////////////////////////////////////////////////////////////////

TEST(OVClassBasicPropsTest, smoke_SetConfigInferenceNumThreads) {
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

TEST(OVClassBasicPropsTest, smoke_SetConfigStreamsNum) {
    ov::Core ie;
    int32_t value = 0;
    int32_t num_streams = 1;

    auto setGetProperty = [&ie](int32_t& getProperty, int32_t setProperty) {
        OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::num_streams(setProperty)));
        OV_ASSERT_NO_THROW(getProperty = ie.get_property("CPU", ov::num_streams));
    };

    setGetProperty(value, num_streams);
    ASSERT_EQ(num_streams, value);

    num_streams = ov::streams::NUMA;

    setGetProperty(value, num_streams);
    ASSERT_GT(value, 0);  // value has been configured automatically

    num_streams = ov::streams::AUTO;

    setGetProperty(value, num_streams);
    ASSERT_GT(value, 0);  // value has been configured automatically
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAffinity) {
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

    const ov::Affinity affinity =
        defaultBindThreadParameter == ov::Affinity::HYBRID_AWARE ? ov::Affinity::NUMA : ov::Affinity::HYBRID_AWARE;
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::affinity(affinity)));
    OV_ASSERT_NO_THROW(value = ie.get_property("CPU", ov::affinity));
    ASSERT_EQ(affinity, value);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigHintInferencePrecision) {
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

TEST(OVClassBasicPropsTest, smoke_SetConfigEnableProfiling) {
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

const auto bf16_if_supported = InferenceEngine::with_cpu_x86_bfloat16() ? ov::element::bf16 : ov::element::f32;
const auto bf16_if_can_be_emulated = InferenceEngine::with_cpu_x86_avx512_core() ? ov::element::bf16 : ov::element::f32;
using ExpectedModeAndType = std::pair<ov::hint::ExecutionMode, ov::element::Type>;

const std::map<ov::hint::ExecutionMode, ExpectedModeAndType> exectedTypeByMode{
    {ov::hint::ExecutionMode::PERFORMANCE, {ov::hint::ExecutionMode::PERFORMANCE, bf16_if_supported}},
    {ov::hint::ExecutionMode::ACCURACY, {ov::hint::ExecutionMode::ACCURACY, ov::element::f32}},
};

TEST(OVClassBasicPropsTest, smoke_SetConfigExecutionModeExpectCorrespondingInferencePrecision) {
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

TEST(OVClassBasicPropsTest, smoke_SetConfigExecutionModeAndInferencePrecision) {
    ov::Core ie;
    const auto inference_precision_default = bf16_if_supported;
    const auto execution_mode_default = ov::hint::ExecutionMode::PERFORMANCE;

    auto expect_execution_mode = [&](const ov::hint::ExecutionMode expected_value) {
        auto execution_mode_value = ov::hint::ExecutionMode::ACCURACY;
        OV_ASSERT_NO_THROW(execution_mode_value = ie.get_property("CPU", ov::hint::execution_mode));
        ASSERT_EQ(execution_mode_value, expected_value);
    };

    auto expect_inference_precision = [&](const ov::element::Type expected_value) {
        auto inference_precision_value = ov::element::undefined;
        OV_ASSERT_NO_THROW(inference_precision_value = ie.get_property("CPU", ov::hint::inference_precision));
        ASSERT_EQ(inference_precision_value, expected_value);
    };

    // check default values
    expect_execution_mode(execution_mode_default);
    expect_inference_precision(inference_precision_default);
    // verify that conflicting property values work as expect
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)));
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));
    expect_execution_mode(
        ov::hint::ExecutionMode::PERFORMANCE);  // inference_preicision does not affect execution_mode property itself
    expect_inference_precision(ov::element::f32);  // inference_preicision has more priority than performance mode

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)));
    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(bf16_if_can_be_emulated)));
    expect_execution_mode(ov::hint::ExecutionMode::ACCURACY);
    expect_inference_precision(bf16_if_can_be_emulated);
}

const std::vector<ov::AnyMap> cpu_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_CPU),
                                            ::testing::ValuesIn(cpu_properties)),
                         OVPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> multi_Auto_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::device_bind_buffer("YES")},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::device_bind_buffer("NO")},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::enable_startup_fallback("YES")},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::intel_auto::enable_startup_fallback("NO")}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiBehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI),
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
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::HIGH)}};
const std::vector<ov::AnyMap> multi_compileModel_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)}};

INSTANTIATE_TEST_SUITE_P(smoke_MultiCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_setcore_properties),
                                            ::testing::ValuesIn(multi_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_setcore_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::HIGH)},
};
const std::vector<ov::AnyMap> auto_compileModel_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::model_priority(ov::hint::Priority::MEDIUM)}};
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
        {ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)},
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
    std::make_pair(ov::AnyMap{{ov::device::priorities(CommonTestUtils::DEVICE_CPU)}}, "CPU")};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiCompileModelBehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI,
                                                              CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(automultiExeDeviceConfigs)),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_multi_device_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU), ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties("CPU", ov::num_streams(4), ov::enable_profiling(true))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{{ov::num_streams(4), ov::enable_profiling(true)}}}})}};

const std::vector<ov::AnyMap> auto_multi_incorrect_device_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::num_streams(4),
     ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::priorities(CommonTestUtils::DEVICE_CPU),
     ov::num_streams(4),
     ov::device::properties("CPU", ov::num_streams(4), ov::enable_profiling(true))}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsNoThrow,
                         OVSetSupportPropCompileModelWithoutConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI,
                                                              CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(auto_multi_device_properties)),
                         OVSetSupportPropCompileModelWithoutConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsThrow,
                         OVSetUnsupportPropCompileModelWithoutConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI,
                                                              CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(auto_multi_incorrect_device_properties)),
                         OVSetUnsupportPropCompileModelWithoutConfigTests::getTestCaseName);

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest, OVSetEnableHyperThreadingHintConfigTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest, OVSetSchedulingCoreTypeHintConfigTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiHeteroOVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values("MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetConfigTest,
                         OVGetConfigTest_ThrowUnsupported,
                         ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetAvailableDevicesPropsTest,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetConfigTest, OVSetEnableCpuPinningHintConfigTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVSetModelPriorityConfigTest,
                         OVSetModelPriorityConfigTest,
                         ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVSetLogLevelConfigTest, OVSetLogLevelConfigTest, ::testing::Values("MULTI", "AUTO"));

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};

const std::vector<ov::AnyMap> configsDeviceProperties = {
    {ov::device::properties("CPU", ov::num_streams(3))},
    {ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(3)}}})}};

const std::vector<ov::AnyMap> configsDevicePropertiesDouble = {
    {ov::device::properties("CPU", ov::num_streams(5)), ov::num_streams(3)},
    {ov::device::properties("CPU", ov::num_streams(5)),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{ov::num_streams(7)}}}),
     ov::num_streams(3)},
    {ov::device::properties("CPU", ov::num_streams(3)), ov::device::properties("CPU", ov::num_streams(5))},
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

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigPropsTest,
                         OVClassSetDevicePriorityConfigPropsTest,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO", "HETERO"),
                                            ::testing::ValuesIn(multiConfigs)));

// IE Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_CPUOVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU", "AUTO:CPU", "MULTI:CPU", "HETERO:CPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_Multi_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(multiConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_AUTO_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_HETERO_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("HETERO"),
                                            ::testing::ValuesIn(heteroConfigsWithSecondaryProperties)));

// IE Class load and check network with ov::device::properties
INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsDeviceProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_CPU_OVClassCompileModelAndCheckWithSecondaryPropertiesDoubleTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values("CPU"),
                                            ::testing::ValuesIn(configsDevicePropertiesDouble)));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(smoke_OVGetConfigTest, OVGetConfigTest, ::testing::Values("CPU"));

// IE Class load and check network with ov::device::properties

}  // namespace
