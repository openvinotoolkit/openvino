// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/properties.hpp"

#include <gtest/gtest.h>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "utils/properties_test.hpp"

#if defined(_WIN32)
#    include <windows.h>
#endif

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
        RO_property(ov::inference_num_threads.name()),
        RO_property(ov::enable_profiling.name()),
        RO_property(ov::hint::inference_precision.name()),
        RO_property(ov::hint::performance_mode.name()),
        RO_property(ov::hint::execution_mode.name()),
        RO_property(ov::hint::num_requests.name()),
        RO_property(ov::hint::enable_cpu_pinning.name()),
        RO_property(ov::hint::enable_cpu_reservation.name()),
        RO_property(ov::hint::scheduling_core_type.name()),
        RO_property(ov::hint::model_distribution_policy.name()),
        RO_property(ov::hint::enable_hyper_threading.name()),
        RO_property(ov::execution_devices.name()),
        RO_property(ov::intel_cpu::denormals_optimization.name()),
        RO_property(ov::log::level.name()),
        RO_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
        RO_property(ov::hint::dynamic_quantization_group_size.name()),
        RO_property(ov::hint::kv_cache_precision.name()),
        RO_property(ov::key_cache_precision.name()),
        RO_property(ov::value_cache_precision.name()),
        RO_property(ov::key_cache_group_size.name()),
        RO_property(ov::value_cache_group_size.name()),
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
    int32_t streams = 1;  // throughput hint should apply higher number of streams
    int32_t value = 0;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::num_streams(streams)));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)));

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCoreStreamsHasHigherPriorityThanLatencyHint) {
    ov::Core ie;
    int32_t streams = ov::get_number_of_cpu_cores();  // latency hint should apply lower number of streams
    int32_t value = 0;

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::num_streams(streams)));
    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);
    OV_ASSERT_NO_THROW(value = compiledModel.get_property(ov::num_streams));
    ASSERT_EQ(streams, value);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckModelStreamsHasHigherPriorityThanLatencyHint) {
    ov::Core ie;
    int32_t streams = ov::get_number_of_cpu_cores();  // latency hint should apply lower number of streams
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
    int32_t streams = 1;  // throughput hint should apply higher number of streams
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

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCpuReservation) {
    ov::Core ie;
    int32_t threads = 1;
    int32_t res_threads = -1;
    bool cpu_reservation = true;
    bool res_cpu_reservation = false;
    bool cpu_pinning = false;
    bool res_cpu_pinning = false;

#if defined(__APPLE__)
    cpu_reservation = false;
    cpu_pinning = false;
#elif defined(__linux__)
    cpu_pinning = true;
#elif defined(_WIN32)
    ULONG highestNodeNumber = 0;
    if (!GetNumaHighestNodeNumber(&highestNodeNumber)) {
        std::cout << "Error getting highest NUMA node number: " << GetLastError() << std::endl;
        return;
    }
    if (highestNodeNumber > 0) {
        cpu_pinning = false;
    } else {
        cpu_pinning = true;
    }
#endif

    OV_ASSERT_NO_THROW(ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)));

    ov::AnyMap config = {{ov::inference_num_threads.name(), threads}, {ov::hint::enable_cpu_reservation.name(), true}};
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(res_threads = compiledModel.get_property(ov::inference_num_threads));
    OV_ASSERT_NO_THROW(res_cpu_reservation = compiledModel.get_property(ov::hint::enable_cpu_reservation));
    OV_ASSERT_NO_THROW(res_cpu_pinning = compiledModel.get_property(ov::hint::enable_cpu_pinning));

    ASSERT_EQ(res_threads, threads);
    ASSERT_EQ(res_cpu_reservation, cpu_reservation);
    ASSERT_EQ(res_cpu_pinning, cpu_pinning);
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

    auto kv_cache_precision_value = ov::element::dynamic;
    OV_ASSERT_NO_THROW(kv_cache_precision_value = compiledModel.get_property(ov::hint::kv_cache_precision));
    ASSERT_EQ(kv_cache_precision_value, ov::element::f32);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkFinetuneKVCachePrecision) {
    ov::Core core;

    core.set_property(deviceName, ov::key_cache_precision(ov::element::f16));
    core.set_property(deviceName, ov::value_cache_precision(ov::element::u4));
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);

    auto key_cache_precision_value = ov::element::dynamic;
    auto value_cache_precision_value = ov::element::dynamic;
    OV_ASSERT_NO_THROW(key_cache_precision_value = compiledModel.get_property(ov::key_cache_precision));
    OV_ASSERT_NO_THROW(value_cache_precision_value = compiledModel.get_property(ov::value_cache_precision));
    ASSERT_EQ(key_cache_precision_value, ov::element::f16);
    ASSERT_EQ(value_cache_precision_value, ov::element::u4);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkFinetuneKVCacheGroupSize) {
    ov::Core core;

    core.set_property(deviceName, ov::key_cache_group_size(32));
    core.set_property(deviceName, ov::value_cache_group_size(16));
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);

    auto key_cache_group_size_value = 0;
    auto value_cache_group_size_value = 0;
    OV_ASSERT_NO_THROW(key_cache_group_size_value = compiledModel.get_property(ov::key_cache_group_size));
    OV_ASSERT_NO_THROW(value_cache_group_size_value = compiledModel.get_property(ov::value_cache_group_size));
    ASSERT_EQ(key_cache_group_size_value, 32);
    ASSERT_EQ(value_cache_group_size_value, 16);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckAccuracyModeDynamicQuantizationGroupSize) {
    ov::Core core;

    ASSERT_NO_THROW(core.set_property(deviceName, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)));
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);

    size_t groupSize = 0;
    ASSERT_NO_THROW(groupSize = compiledModel.get_property(ov::hint::dynamic_quantization_group_size));
    ASSERT_EQ(groupSize, 0);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckAccuracyModeKVCachePrecision) {
    ov::Core core;

    ASSERT_NO_THROW(core.set_property(deviceName, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)));
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);

    auto kv_cache_precision_value = ov::element::dynamic;
    ASSERT_NO_THROW(kv_cache_precision_value = compiledModel.get_property(ov::hint::kv_cache_precision));
    ASSERT_EQ(kv_cache_precision_value, ov::element::f32);
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

TEST_F(OVClassConfigTestCPU,
       smoke_CpuExecNetworkCheckModelInferencePrecisionHasHigherPriorityThanCoreInferencePrecision) {
    ov::Core ie;
    auto inference_precision_value = ov::element::dynamic;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));

    ov::AnyMap config;
    config[ov::hint::inference_precision.name()] = bf16_if_can_be_emulated;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(inference_precision_value = compiledModel.get_property(ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, bf16_if_can_be_emulated);
}

TEST_F(OVClassConfigTestCPU,
       smoke_CpuExecNetworkCheckCoreInferencePrecisionHasHigherPriorityThanModelPerformanceExecutionMode) {
    ov::Core ie;
    auto execution_mode_value = ov::hint::ExecutionMode::ACCURACY;
    auto inference_precision_value = ov::element::dynamic;

    OV_ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));

    ov::AnyMap config;
    config[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName, config);

    OV_ASSERT_NO_THROW(execution_mode_value = compiledModel.get_property(ov::hint::execution_mode));
    ASSERT_EQ(execution_mode_value, ov::hint::ExecutionMode::PERFORMANCE);

    OV_ASSERT_NO_THROW(inference_precision_value = compiledModel.get_property(ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, ov::element::f32);
}

TEST_F(OVClassConfigTestCPU,
       smoke_CpuExecNetworkCheckModelInferencePrecisionHasHigherPriorityThanCorePerformanceExecutionMode) {
    ov::Core ie;
    auto execution_mode_value = ov::hint::ExecutionMode::PERFORMANCE;
    auto inference_precision_value = ov::element::dynamic;
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
    // check set and get
    const std::vector<ov::log::Level> logLevels = {ov::log::Level::ERR,
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

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCPURuntimOptions) {
    ov::Core ie;
    ov::Any type;
    ov::Any size;
    ov::Any keySize;
    ov::Any valueSize;
    ov::Any keyCacheType;
    ov::Any valueCacheType;
    ov::CompiledModel compiledModel;
    model->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
    model->set_rt_info("0", "runtime_options", ov::hint::dynamic_quantization_group_size.name());
    model->set_rt_info("32", "runtime_options", ov::key_cache_group_size.name());
    model->set_rt_info("16", "runtime_options", ov::value_cache_group_size.name());
    model->set_rt_info("u8", "runtime_options", ov::key_cache_precision.name());
    model->set_rt_info("u8", "runtime_options", ov::value_cache_precision.name());
    OV_ASSERT_NO_THROW(compiledModel = ie.compile_model(model, deviceName));
    OV_ASSERT_NO_THROW(type = compiledModel.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(size = compiledModel.get_property(ov::hint::dynamic_quantization_group_size));
    OV_ASSERT_NO_THROW(keySize = compiledModel.get_property(ov::key_cache_group_size));
    OV_ASSERT_NO_THROW(valueSize = compiledModel.get_property(ov::value_cache_group_size));
    OV_ASSERT_NO_THROW(keyCacheType = compiledModel.get_property(ov::key_cache_precision));
    OV_ASSERT_NO_THROW(valueCacheType = compiledModel.get_property(ov::value_cache_precision));
    ASSERT_EQ(type.as<ov::element::Type>(), ov::element::f16);
    ASSERT_EQ(size.as<uint64_t>(), 0);
    ASSERT_EQ(keySize.as<uint64_t>(), 32);
    ASSERT_EQ(valueSize.as<uint64_t>(), 16);
    ASSERT_EQ(keyCacheType.as<ov::element::Type>(), ov::element::u8);
    ASSERT_EQ(valueCacheType.as<ov::element::Type>(), ov::element::u8);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCPURuntimOptionsWithCompileConfig) {
    ov::Core ie;
    ov::Any type;
    ov::Any size;
    ov::Any keySize;
    ov::Any valueSize;
    ov::Any keyCacheType;
    ov::Any valueCacheType;
    ov::CompiledModel compiledModel;
    model->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
    model->set_rt_info("0", "runtime_options", ov::hint::dynamic_quantization_group_size.name());
    model->set_rt_info("0", "runtime_options", ov::key_cache_group_size.name());
    model->set_rt_info("0", "runtime_options", ov::value_cache_group_size.name());
    model->set_rt_info("f32", "runtime_options", ov::key_cache_precision.name());
    model->set_rt_info("f32", "runtime_options", ov::value_cache_precision.name());
    ov::AnyMap config;
    config[ov::hint::kv_cache_precision.name()] = "u8";
    config[ov::hint::dynamic_quantization_group_size.name()] = "16";
    // propperty has higher priority than rt_info
    config[ov::key_cache_group_size.name()] = "32";
    config[ov::value_cache_group_size.name()] = "16";
    // key/value cache prec has higher priority than kvCachePrec
    config[ov::key_cache_precision.name()] = "f16";
    config[ov::value_cache_precision.name()] = "bf16";
    OV_ASSERT_NO_THROW(compiledModel = ie.compile_model(model, deviceName, config));
    OV_ASSERT_NO_THROW(type = compiledModel.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(size = compiledModel.get_property(ov::hint::dynamic_quantization_group_size));
    OV_ASSERT_NO_THROW(keySize = compiledModel.get_property(ov::key_cache_group_size));
    OV_ASSERT_NO_THROW(valueSize = compiledModel.get_property(ov::value_cache_group_size));
    OV_ASSERT_NO_THROW(keyCacheType = compiledModel.get_property(ov::key_cache_precision));
    OV_ASSERT_NO_THROW(valueCacheType = compiledModel.get_property(ov::value_cache_precision));
    ASSERT_EQ(type.as<ov::element::Type>(), ov::element::u8);
    ASSERT_EQ(size.as<uint64_t>(), 16);
    ASSERT_EQ(keySize.as<uint64_t>(), 32);
    ASSERT_EQ(valueSize.as<uint64_t>(), 16);
    ASSERT_EQ(keyCacheType.as<ov::element::Type>(), ov::element::f16);
    ASSERT_EQ(valueCacheType.as<ov::element::Type>(), ov::element::bf16);
}

TEST_F(OVClassConfigTestCPU, smoke_CpuExecNetworkCheckCPURuntimOptionsWithCoreProperties) {
    ov::Core core;
    ov::Any type;
    ov::Any size;
    ov::Any keySize;
    ov::Any valueSize;
    ov::Any keyCacheType;
    ov::Any valueCacheType;
    core.set_property(deviceName, ov::hint::kv_cache_precision(ov::element::f32));
    core.set_property(deviceName, ov::hint::dynamic_quantization_group_size(16));
    core.set_property(deviceName, ov::key_cache_group_size(8));
    core.set_property(deviceName, ov::value_cache_group_size(8));
    core.set_property(deviceName, ov::key_cache_precision(ov::element::f16));
    core.set_property(deviceName, ov::value_cache_precision(ov::element::bf16));

    ov::CompiledModel compiledModel;
    model->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
    model->set_rt_info("0", "runtime_options", ov::hint::dynamic_quantization_group_size.name());
    model->set_rt_info("32", "runtime_options", ov::key_cache_group_size.name());
    model->set_rt_info("16", "runtime_options", ov::value_cache_group_size.name());
    // User's setting has higher priority than rt_info
    model->set_rt_info("f32", "runtime_options", ov::key_cache_precision.name());
    model->set_rt_info("f32", "runtime_options", ov::value_cache_precision.name());

    OV_ASSERT_NO_THROW(compiledModel = core.compile_model(model, deviceName));
    OV_ASSERT_NO_THROW(type = compiledModel.get_property(ov::hint::kv_cache_precision));
    OV_ASSERT_NO_THROW(size = compiledModel.get_property(ov::hint::dynamic_quantization_group_size));
    OV_ASSERT_NO_THROW(keySize = compiledModel.get_property(ov::key_cache_group_size));
    OV_ASSERT_NO_THROW(valueSize = compiledModel.get_property(ov::value_cache_group_size));
    OV_ASSERT_NO_THROW(keyCacheType = compiledModel.get_property(ov::key_cache_precision));
    OV_ASSERT_NO_THROW(valueCacheType = compiledModel.get_property(ov::value_cache_precision));

    ASSERT_EQ(type.as<ov::element::Type>(), ov::element::f32);
    ASSERT_EQ(size.as<uint64_t>(), 16);
    ASSERT_EQ(keySize.as<uint64_t>(), 8);
    ASSERT_EQ(valueSize.as<uint64_t>(), 8);
    ASSERT_EQ(keyCacheType.as<ov::element::Type>(), ov::element::f16);
    ASSERT_EQ(valueCacheType.as<ov::element::Type>(), ov::element::bf16);
}

}  // namespace
