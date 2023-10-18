// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "test_utils/properties_test.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_system_conf.h"

#include <algorithm>

namespace {

TEST_F(OVClassConfigTestCPU, smoke_PluginAllSupportedPropertiesAreAvailable) {
    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };
    auto RW_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
    };

    std::vector<ov::PropertyName> expectedSupportedProperties{
        // read only
        RO_property(ov::supported_properties.name()),
        RO_property(ov::available_devices.name()),
        RO_property(ov::range_for_async_infer_requests.name()),
        RO_property(ov::range_for_streams.name()),
        RO_property(ov::device::full_name.name()),
        RO_property(ov::device::capabilities.name()),
        // read write
        RW_property(ov::num_streams.name()),
        RW_property(ov::affinity.name()),
        RW_property(ov::inference_num_threads.name()),
        RW_property(ov::enable_profiling.name()),
        RW_property(ov::hint::inference_precision.name()),
        RW_property(ov::hint::performance_mode.name()),
        RW_property(ov::hint::execution_mode.name()),
        RW_property(ov::hint::num_requests.name()),
        RW_property(ov::hint::enable_cpu_pinning.name()),
        RW_property(ov::hint::scheduling_core_type.name()),
        RW_property(ov::hint::enable_hyper_threading.name()),
        RW_property(ov::device::id.name()),
        RW_property(ov::intel_cpu::denormals_optimization.name()),
        RW_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
    };

    ov::Core ie;
    std::vector<ov::PropertyName> supportedProperties;
    ASSERT_NO_THROW(supportedProperties = ie.get_property("CPU", ov::supported_properties));
    // the order of supported properties does not matter, sort to simplify the comparison
    std::sort(expectedSupportedProperties.begin(), expectedSupportedProperties.end());
    std::sort(supportedProperties.begin(), supportedProperties.end());

    ASSERT_EQ(supportedProperties, expectedSupportedProperties);
}

TEST_F(OVClassConfigTestCPU, smoke_PluginGetPropertiesDoesNotThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ASSERT_NO_THROW(properties = ie.get_property("CPU", ov::supported_properties));

    for (const auto& property : properties) {
        ASSERT_NO_THROW((void)ie.get_property("CPU", property));
    }
}

TEST_F(OVClassConfigTestCPU, smoke_PluginSetROPropertiesThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ASSERT_NO_THROW(properties = ie.get_property("CPU", ov::supported_properties));

    for (const auto& property : properties) {
        if (property.is_mutable())
            continue;

        ASSERT_THROW(ie.set_property("CPU", {{property, "DUMMY VALUE"}}), ov::Exception);
    }
}

TEST_F(OVClassConfigTestCPU, smoke_PluginSetConfigInferenceNumThreads) {
    ov::Core ie;
    int32_t value = 0;
    int32_t num_threads = 1;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::inference_num_threads(num_threads)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::inference_num_threads));
    ASSERT_EQ(num_threads, value);

    num_threads = 4;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::inference_num_threads(num_threads)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::inference_num_threads));
    ASSERT_EQ(num_threads, value);
}

TEST_F(OVClassConfigTestCPU, smoke_PluginSetConfigStreamsNum) {
    ov::Core ie;
    int32_t value = 0;
    int32_t num_streams = 1;

    auto setGetProperty = [&ie](int32_t& getProperty, int32_t setProperty){
        ASSERT_NO_THROW(ie.set_property("CPU", ov::num_streams(setProperty)));
        ASSERT_NO_THROW(getProperty = ie.get_property("CPU", ov::num_streams));
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

TEST_F(OVClassConfigTestCPU, smoke_PluginSetConfigAffinity) {
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
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::affinity));
    ASSERT_EQ(defaultBindThreadParameter, value);

    const ov::Affinity affinity = defaultBindThreadParameter == ov::Affinity::HYBRID_AWARE ? ov::Affinity::NUMA : ov::Affinity::HYBRID_AWARE;
    ASSERT_NO_THROW(ie.set_property("CPU", ov::affinity(affinity)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::affinity));
    ASSERT_EQ(affinity, value);
}

TEST_F(OVClassConfigTestCPU, smoke_PluginSetConfigAffinityCore) {
    ov::Core ie;
    ov::Affinity affinity = ov::Affinity::CORE;
    bool value = false;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::affinity(affinity)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::enable_cpu_pinning));
    ASSERT_EQ(true, value);

    affinity = ov::Affinity::HYBRID_AWARE;
    ASSERT_NO_THROW(ie.set_property("CPU", ov::affinity(affinity)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::enable_cpu_pinning));
    ASSERT_EQ(true, value);

    affinity = ov::Affinity::NUMA;
    ASSERT_NO_THROW(ie.set_property("CPU", ov::affinity(affinity)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::enable_cpu_pinning));
    ASSERT_EQ(false, value);
}

#if defined(OV_CPU_ARM_ENABLE_FP16)
    const auto expected_precision_for_performance_mode = ov::element::f16;
#else
    const auto expected_precision_for_performance_mode = InferenceEngine::with_cpu_x86_bfloat16() ? ov::element::bf16 : ov::element::f32;
#endif

TEST_F(OVClassConfigTestCPU, smoke_PluginSetConfigHintInferencePrecision) {
    ov::Core ie;
    auto value = ov::element::f32;

    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(expected_precision_for_performance_mode, value);

    const auto forcedPrecision = ov::element::f32;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(forcedPrecision)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(value, forcedPrecision);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto forced_precision_deprecated = ov::element::f32;
    ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(forced_precision_deprecated)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(value, forced_precision_deprecated);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST_F(OVClassConfigTestCPU, smoke_PluginSetConfigEnableProfiling) {
    ov::Core ie;
    auto value = false;
    const bool enableProfilingDefault = false;

    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::enable_profiling));
    ASSERT_EQ(enableProfilingDefault, value);

    const bool enableProfiling = true;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::enable_profiling(enableProfiling)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::enable_profiling));
    ASSERT_EQ(enableProfiling, value);
}

const auto bf16_if_can_be_emulated = InferenceEngine::with_cpu_x86_avx512_core() ? ov::element::bf16 : ov::element::f32;
using ExpectedModeAndType = std::pair<ov::hint::ExecutionMode, ov::element::Type>;

const std::map<ov::hint::ExecutionMode, ExpectedModeAndType> expectedTypeByMode {
    {ov::hint::ExecutionMode::PERFORMANCE, {ov::hint::ExecutionMode::PERFORMANCE,
                                            expected_precision_for_performance_mode}},
    {ov::hint::ExecutionMode::ACCURACY,    {ov::hint::ExecutionMode::ACCURACY,
                                            ov::element::f32}},
};

TEST_F(OVClassConfigTestCPU, smoke_PluginSetConfigExecutionModeExpectCorrespondingInferencePrecision) {
    ov::Core ie;
    const auto inference_precision_default = expected_precision_for_performance_mode;
    const auto execution_mode_default = ov::hint::ExecutionMode::PERFORMANCE;
    auto execution_mode_value = ov::hint::ExecutionMode::PERFORMANCE;
    auto inference_precision_value = ov::element::undefined;

    // check default values
    ASSERT_NO_THROW(inference_precision_value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(inference_precision_value, inference_precision_default);
    ASSERT_NO_THROW(execution_mode_value = ie.get_property("CPU", ov::hint::execution_mode));
    ASSERT_EQ(execution_mode_value, execution_mode_default);

    for (const auto& m : expectedTypeByMode) {
        const auto execution_mode = m.first;
        const auto execution_mode_expected = m.second.first;
        const auto inference_precision_expected = m.second.second;

        ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(execution_mode)));
        ASSERT_NO_THROW(execution_mode_value = ie.get_property("CPU", ov::hint::execution_mode));
        ASSERT_EQ(execution_mode_value, execution_mode_expected);

        ASSERT_NO_THROW(inference_precision_value = ie.get_property("CPU", ov::hint::inference_precision));
        ASSERT_EQ(inference_precision_value, inference_precision_expected);
    }
}

TEST_F(OVClassConfigTestCPU, smoke_PluginSetConfigExecutionModeAndInferencePrecision) {
    ov::Core ie;
    const auto inference_precision_default = expected_precision_for_performance_mode;
    const auto execution_mode_default = ov::hint::ExecutionMode::PERFORMANCE;

    auto expect_execution_mode = [&](const ov::hint::ExecutionMode expected_value) {
        auto execution_mode_value = ov::hint::ExecutionMode::ACCURACY;
        ASSERT_NO_THROW(execution_mode_value = ie.get_property("CPU", ov::hint::execution_mode));
        ASSERT_EQ(execution_mode_value, expected_value);
    };

    auto expect_inference_precision = [&](const ov::element::Type expected_value) {
        auto inference_precision_value = ov::element::undefined;;
        ASSERT_NO_THROW(inference_precision_value = ie.get_property("CPU", ov::hint::inference_precision));
        ASSERT_EQ(inference_precision_value, expected_value);
    };

    // check default values
    expect_execution_mode(execution_mode_default);
    expect_inference_precision(inference_precision_default);
    // verify that conflicting property values work as expect
    ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)));
    ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(ov::element::f32)));
    expect_execution_mode(ov::hint::ExecutionMode::PERFORMANCE); // inference_preicision does not affect execution_mode property itself
    expect_inference_precision(ov::element::f32); // inference_preicision has more priority than performance mode

    ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)));
    ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(bf16_if_can_be_emulated)));
    expect_execution_mode(ov::hint::ExecutionMode::ACCURACY);
    expect_inference_precision(bf16_if_can_be_emulated);
}

} // namespace
