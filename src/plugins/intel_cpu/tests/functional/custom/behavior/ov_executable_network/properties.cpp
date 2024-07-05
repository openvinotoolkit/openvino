// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils/properties_test.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"

namespace {

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkSupportedPropertiesAreAvailable) {
    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };

    std::vector<ov::PropertyName> expectedSupportedProperties{
        // read only
        RO_property(ov::supported_properties.name()),
        RO_property(ov::model_name.name()),
        RO_property(ov::optimal_number_of_infer_requests.name()),
        RO_property(ov::num_streams.name()),
        RO_property(ov::affinity.name()),
        RO_property(ov::inference_num_threads.name()),
        RO_property(ov::enable_profiling.name()),
        RO_property(ov::hint::inference_precision.name()),
        RO_property(ov::hint::performance_mode.name()),
        RO_property(ov::hint::execution_mode.name()),
        RO_property(ov::hint::num_requests.name()),
        RO_property(ov::hint::enable_cpu_pinning.name()),
        RO_property(ov::hint::scheduling_core_type.name()),
        RO_property(ov::hint::model_distribution_policy.name()),
        RO_property(ov::hint::enable_hyper_threading.name()),
        RO_property(ov::execution_devices.name()),
        RO_property(ov::intel_cpu::denormals_optimization.name()),
        RO_property(ov::log::level.name()),
        RO_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
        RO_property(ov::hint::dynamic_quantization_group_size.name()),
        RO_property(ov::hint::kv_cache_precision.name()),
    };

    ov::Core ie;
    std::vector<ov::PropertyName> supportedProperties;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    OV_ASSERT_NO_THROW(supportedProperties = compiledModel.get_property(ov::supported_properties));
    // the order of supported properties does not matter, sort to simplify the comparison
    std::sort(expectedSupportedProperties.begin(), expectedSupportedProperties.end());
    std::sort(supportedProperties.begin(), supportedProperties.end());

    ASSERT_EQ(supportedProperties, expectedSupportedProperties);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkGetROPropertiesDoesNotThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);

    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));

    for (const auto& property : properties) {
        OV_ASSERT_NO_THROW((void)compiledModel.get_property(property));
    }
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkSetROPropertiesThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);

    OV_ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));

    for (auto it = properties.begin(); it != properties.end(); ++it) {
        ASSERT_TRUE(it != properties.end());
        ASSERT_FALSE(it->is_mutable());
        ASSERT_THROW(compiledModel.set_property({{*it, "DUMMY VALUE"}}), ov::Exception);
    }
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCoreStreamsHasHigherPriorityThanThroughputHint) {
    ov::Core ie;
    int32_t streams = 1; // throughput hint should apply higher number of streams
    int32_t value = 0;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::num_streams(streams)));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)));

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCoreStreamsHasHigherPriorityThanLatencyHint) {
    ov::Core ie;
    int32_t streams = ov::get_number_of_cpu_cores(); // latency hint should apply lower number of streams
    int32_t value = 0;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::num_streams(streams)));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckModelStreamsHasHigherPriorityThanLatencyHint) {
    ov::Core ie;
    int32_t streams = ov::get_number_of_cpu_cores(); // latency hint should apply lower number of streams
    int32_t value = 0;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::AnyMap config;
    config[ov::num_streams.name()] = streams;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckModelStreamsHasHigherPriorityThanThroughputHint) {
    ov::Core ie;
    int32_t streams = 1; // throughput hint should apply higher number of streams
    int32_t value = 0;

    ov::AnyMap config;
    config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    config[ov::num_streams.name()] = streams;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckModelZeroStreams) {
    ov::Core ie;
    int32_t streams = 0;
    int32_t value = -1;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::AnyMap config;
    config[ov::num_streams.name()] = streams;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));

    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckSparseWeigthsDecompressionRate) {
    ov::Core core;

    core.set_property(deviceName, ov::intel_cpu::sparse_weights_decompression_rate(0.8));
    OV_ASSERT_NO_THROW(ov::CompiledModel compiledModel = core.compile_model(model, deviceName));
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckDynamicQuantizationGroupSize) {
    ov::Core core;

    core.set_property(deviceName, ov::hint::dynamic_quantization_group_size(64));
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);

    size_t groupSize = 0;
    OV_ASSERT_NO_THROW(groupSize = compiledModel.get_property(ov::hint::dynamic_quantization_group_size));
    ASSERT_EQ(groupSize, 64);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckKVCachePrecision) {
    ov::Core core;

    core.set_property(deviceName, ov::hint::kv_cache_precision(ov::element::f32));
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);

    auto kv_cache_precision_value = ov::element::undefined;
    OV_ASSERT_NO_THROW(kv_cache_precision_value = compiledModel.get_property(ov::hint::kv_cache_precision));
    ASSERT_EQ(kv_cache_precision_value, ov::element::f32);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckAccuracyModeDynamicQuantizationGroupSize) {
    ov::Core core;

    ASSERT_NO_THROW(core.set_property(deviceName, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)));
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);

    size_t groupSize = 0;
    ASSERT_NO_THROW(groupSize = compiledModel.get_property(ov::hint::dynamic_quantization_group_size));
    ASSERT_EQ(groupSize, 0);
}

const auto bf16_if_can_be_emulated = ov::with_cpu_x86_avx512_core() ? ov::element::bf16 : ov::element::f32;

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckExecutionModeIsAvailableInCoreAndModel) {
    ov::Core ie;
    std::vector<ov::PropertyName> ie_properties;

    OV_ASSERT_NO_THROW(ie_properties = ie.get_property(deviceName, ov::supported_properties));
    const auto ie_exec_mode_it = find(ie_properties.begin(), ie_properties.end(), ov::hint::execution_mode);
    ASSERT_NE(ie_exec_mode_it, ie_properties.end());
    ASSERT_TRUE(ie_exec_mode_it->is_mutable());

    ov::AnyMap config;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);
    std::vector<ov::PropertyName> model_properties;

    OV_ASSERT_NO_THROW(model_properties = compiledModel.get_property(ov::supported_properties));
    const auto model_exec_mode_it = find(model_properties.begin(), model_properties.end(), ov::hint::execution_mode);
    ASSERT_NE(model_exec_mode_it, model_properties.end());
    ASSERT_FALSE(model_exec_mode_it->is_mutable());
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckModelInferencePrecisionHasHigherPriorityThanCoreInferencePrecision) {
    ov::Core ie;
    auto inference_precision_value = ov::element::undefined;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));

    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = bf16_if_can_be_emulated;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(inference_precision_value = compiledModel.get_property(ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, bf16_if_can_be_emulated);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCoreInferencePrecisionHasHigherPriorityThanModelPerformanceExecutionMode) {
    ov::Core ie;
    auto execution_mode_value = ov::hint::ExecutionMode::ACCURACY;
    auto inference_precision_value = ov::element::undefined;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));

    ov::AnyMap config;
    config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(execution_mode_value = compiledModel.get_property(ov::hint::execution_mode));
    ASSERT_EQ(execution_mode_value, ov::hint::ExecutionMode::PERFORMANCE);

    OV_ASSERT_NO_THROW(inference_precision_value = compiledModel.get_property(ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, ov::element::f32);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckModelInferencePrecisionHasHigherPriorityThanCorePerformanceExecutionMode) {
    ov::Core ie;
    auto execution_mode_value = ov::hint::ExecutionMode::PERFORMANCE;
    auto inference_precision_value = ov::element::undefined;
    const auto inference_precision_expected = bf16_if_can_be_emulated;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)));

    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = inference_precision_expected;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(execution_mode_value = compiledModel.get_property(ov::hint::execution_mode));
    ASSERT_EQ(execution_mode_value, ov::hint::ExecutionMode::ACCURACY);

    OV_ASSERT_NO_THROW(inference_precision_value = compiledModel.get_property(ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, inference_precision_expected);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckLogLevel) {
    ov::Core ie;

    // check default value
    {
        ov::AnyMap config;
        ov::Any value;
        ov::CompiledModel compiledModel;
        OV_ASSERT_NO_THROW(compiledModel = ie.compile_model(model, deviceName, config));
        OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::log::level));
        ASSERT_EQ(value.as<ov::log::Level>(), ov::log::Level::NO);
    }
    //check set and get
    const std::vector<ov::log::Level> logLevels = {
        ov::log::Level::ERR,
        ov::log::Level::NO,
        ov::log::Level::WARNING,
        ov::log::Level::INFO,
        ov::log::Level::DEBUG,
        ov::log::Level::TRACE};

    for (unsigned int i = 0; i < logLevels.size(); i++) {
        ov::Any value;
        ov::CompiledModel compiledModel;
        ov::AnyMap config{ov::log::level(logLevels[i])};
        OV_ASSERT_NO_THROW(compiledModel = ie.compile_model(model, deviceName, config));
        OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::log::level));
        ASSERT_EQ(value.as<ov::log::Level>(), logLevels[i]);
    }

    for (unsigned int i = 0; i < logLevels.size(); i++) {
        ov::Any value;
        ov::CompiledModel compiledModel;
        OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::log::level(logLevels[i])));
        OV_ASSERT_NO_THROW(compiledModel = ie.compile_model(model, deviceName));
        OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::log::level));
        ASSERT_EQ(value.as<ov::log::Level>(), logLevels[i]);
    }
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCPUExecutionDevice) {
    ov::Core ie;
    ov::Any value;
    ov::CompiledModel compiledModel;

    OV_ASSERT_NO_THROW(compiledModel = ie.compile_model(model, deviceName));
    OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::execution_devices));
    ASSERT_EQ(value.as<std::string>(), "CPU");
}

} // namespace
