// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include <base/ov_behavior_test_utils.hpp>

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "ie_system_conf.h"

#include <gtest/gtest.h>

using namespace ov::test::behavior;
namespace {

//
// Executable Network GetMetric
//
class OVClassConfigTestCPU : public ::testing::Test {
public:
    std::shared_ptr<ngraph::Function> model;
    const std::string deviceName = "CPU";

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ngraph::builder::subgraph::makeConvPoolRelu();
    }
};

TEST_F(OVClassConfigTestCPU, smoke_GetROPropertiesDoesNotThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);

    ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));

    for (const auto& property : properties) {
        ASSERT_NO_THROW((void)compiledModel.get_property(property));
    }
}

TEST_F(OVClassConfigTestCPU, smoke_SetROPropertiesThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);

    ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));

    for (auto it = properties.begin(); it != properties.end(); ++it) {
        ASSERT_TRUE(it != properties.end());
        ASSERT_FALSE(it->is_mutable());
        ASSERT_THROW(compiledModel.set_property({{*it, "DUMMY VALUE"}}), ov::Exception);
    }
}

TEST_F(OVClassConfigTestCPU, smoke_CheckCoreStreamsHasHigherPriorityThanThroughputHint) {
    ov::Core ie;
    int32_t streams = 1; // throughput hint should apply higher number of streams
    int32_t value = 0;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::num_streams(streams)));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)));

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckCoreStreamsHasHigherPriorityThanLatencyHint) {
    ov::Core ie;
    int32_t streams = 4; // latency hint should apply lower number of streams
    int32_t value = 0;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::num_streams(streams)));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckModelStreamsHasHigherPriorityThanLatencyHints) {
    ov::Core ie;
    int32_t streams = 4; // latency hint should apply lower number of streams
    int32_t value = 0;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::AnyMap config;
    config[ov::num_streams.name()] = streams;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckModelStreamsHasHigherPriorityThanThroughputHint) {
    ov::Core ie;
    int32_t streams = 1; // throughput hint should apply higher number of streams
    int32_t value = 0;

    ov::AnyMap config;
    config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    config[ov::num_streams.name()] = streams;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckSparseWeigthsDecompressionRate) {
    ov::Core core;

    core.set_property(deviceName, ov::intel_cpu::sparse_weights_decompression_rate(0.8));
    ASSERT_NO_THROW(ov::CompiledModel compiledModel = core.compile_model(model, deviceName));
}

const auto bf16_if_can_be_emulated = InferenceEngine::with_cpu_x86_avx512_core() ? ov::element::bf16 : ov::element::f32;

TEST_F(OVClassConfigTestCPU, smoke_CheckExecutionModeIsAvailableInCoreAndModel) {
    ov::Core ie;
    std::vector<ov::PropertyName> ie_properties;

    ASSERT_NO_THROW(ie_properties = ie.get_property(deviceName, ov::supported_properties));
    const auto ie_exec_mode_it = find(ie_properties.begin(), ie_properties.end(), ov::hint::execution_mode);
    ASSERT_NE(ie_exec_mode_it, ie_properties.end());
    ASSERT_TRUE(ie_exec_mode_it->is_mutable());

    ov::AnyMap config;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);
    std::vector<ov::PropertyName> model_properties;

    ASSERT_NO_THROW(model_properties = compiledModel.get_property(ov::supported_properties));
    const auto model_exec_mode_it = find(model_properties.begin(), model_properties.end(), ov::hint::execution_mode);
    ASSERT_NE(model_exec_mode_it, model_properties.end());
    ASSERT_FALSE(model_exec_mode_it->is_mutable());
}

TEST_F(OVClassConfigTestCPU, smoke_CheckModelInferencePrecisionHasHigherPriorityThanCoreInferencePrecision) {
    ov::Core ie;
    auto inference_precision_value = ov::element::undefined;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));

    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = bf16_if_can_be_emulated;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    ASSERT_NO_THROW(inference_precision_value = compiledModel.get_property(ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, bf16_if_can_be_emulated);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckCoreInferencePrecisionHasHigherPriorityThanModelPerformanceExecutionMode) {
    ov::Core ie;
    auto execution_mode_value = ov::hint::ExecutionMode::ACCURACY;
    auto inference_precision_value = ov::element::undefined;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));

    ov::AnyMap config;
    config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    ASSERT_NO_THROW(execution_mode_value = compiledModel.get_property(ov::hint::execution_mode));
    ASSERT_EQ(execution_mode_value, ov::hint::ExecutionMode::PERFORMANCE);

    ASSERT_NO_THROW(inference_precision_value = compiledModel.get_property(ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, ov::element::f32);
}

TEST_F(OVClassConfigTestCPU, smoke_CheckModelInferencePrecisionHasHigherPriorityThanCorePerformanceExecutionMode) {
    ov::Core ie;
    auto execution_mode_value = ov::hint::ExecutionMode::PERFORMANCE;
    auto inference_precision_value = ov::element::undefined;
    const auto inference_precision_expected = bf16_if_can_be_emulated;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)));

    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = inference_precision_expected;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    ASSERT_NO_THROW(execution_mode_value = compiledModel.get_property(ov::hint::execution_mode));
    ASSERT_EQ(execution_mode_value, ov::hint::ExecutionMode::ACCURACY);

    ASSERT_NO_THROW(inference_precision_value = compiledModel.get_property(ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, inference_precision_expected);
}

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)),
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY::getTestCaseName);

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO:CPU"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));

} // namespace
