// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include <openvino/runtime/auto/properties.hpp>
#include <thread>

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#ifdef _WIN32
#    include "gpu/gpu_context_api_dx.hpp"
#elif defined ENABLE_LIBVA
#    include <gpu/gpu_context_api_va.hpp>
#endif
#include "gpu/gpu_config.hpp"
#include "gpu/gpu_context_api_ocl.hpp"

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {

const std::vector<ov::AnyMap> gpu_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(gpu_properties)),
                         OVPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> gpu_setcore_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
     ov::hint::num_requests(2),
     ov::enable_profiling(false)}};
const std::vector<ov::AnyMap> gpu_compileModel_properties = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
     ov::hint::num_requests(10),
     ov::enable_profiling(true)}};

INSTANTIATE_TEST_SUITE_P(smoke_gpuCompileModelBehaviorTests,
                         OVSetPropComplieModleGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(gpu_setcore_properties),
                                            ::testing::ValuesIn(gpu_compileModel_properties)),
                         OVSetPropComplieModleGetPropTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCommon,
                         OVBasicPropertiesTestsP,
                         ::testing::Values(std::make_pair("openvino_intel_gpu_plugin", "GPU")));

// //
// // IE Class GetMetric
// //

INSTANTIATE_TEST_SUITE_P(nightly_HeteroAutoBatchOVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values("HETERO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_gpuOVGetMetricPropsTest, OVGetMetricPropsTest, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVGetAvailableDevicesPropsTest,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(
    smoke_OVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                        { ov::device::uuid.name(), ov::device::luid.name(), ov::device::gops.name(), ov::device::type.name(), ov::device::full_name.name() }))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_HeteroAutoBatchOVCheckChangePropComplieModleGetPropTests_DEVICE_ID,
                         OVCheckChangePropComplieModleGetPropTests_DEVICE_ID,
                         ::testing::Combine(::testing::Values("HETERO", "BATCH"),
                                            ::testing::Values(ov::AnyMap({}))),
                        OVCheckChangePropComplieModleGetPropTests_DEVICE_ID::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_gpuOVCheckChangePropComplieModleGetPropTests_DEVICE_ID,
                         OVCheckChangePropComplieModleGetPropTests_DEVICE_ID,
                         ::testing::Combine(::testing::Values("GPU"),
                                            ::testing::Values(ov::AnyMap({}))),
                        OVCheckChangePropComplieModleGetPropTests_DEVICE_ID::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_OVCheckSetSupportedRWMetricsPropsTests,
    OVCheckSetSupportedRWMetricsPropsTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWMandatoryPropertiesValues(
                           {ov::hint::execution_mode.name()}))),
    OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);

auto multiConfigs = []() {
    return std::vector<ov::AnyMap>{
        {ov::device::priorities(ov::test::utils::DEVICE_CPU)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU)},
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassSetDevicePriorityConfigPropsTest,
                         OVClassSetDevicePriorityConfigPropsTest,
                         ::testing::Combine(::testing::Values("HETERO"),
                                            ::testing::ValuesIn(multiConfigs())));
//
// GPU specific metrics
//
using OVGetMetricPropsTest_GPU_DEVICE_TOTAL_MEM_SIZE = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_DEVICE_TOTAL_MEM_SIZE, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(target_device, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)));
    auto t = p.as<uint64_t>();

    std::cout << "GPU device total memory size: " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVGetMetricPropsTest,
                         OVGetMetricPropsTest_GPU_DEVICE_TOTAL_MEM_SIZE,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_UARCH_VERSION = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_UARCH_VERSION, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(target_device, GPU_METRIC_KEY(UARCH_VERSION)));
    auto t = p.as<std::string>();

    std::cout << "GPU device uarch: " << t << std::endl;
    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(UARCH_VERSION));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVGetMetricPropsTest,
                         OVGetMetricPropsTest_GPU_UARCH_VERSION,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_EXECUTION_UNITS_COUNT = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_EXECUTION_UNITS_COUNT, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(target_device, GPU_METRIC_KEY(EXECUTION_UNITS_COUNT)));
    auto t = p.as<int>();

    std::cout << "GPU EUs count: " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(EXECUTION_UNITS_COUNT));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVGetMetricPropsTest,
                         OVGetMetricPropsTest_GPU_EXECUTION_UNITS_COUNT,
                         ::testing::Values("GPU"));

using OVClassGetPropertyTest_GPU = OVClassBaseTestP;
TEST_P(OVClassGetPropertyTest_GPU, GetMetricAvailableDevicesAndPrintNoThrow) {
    ov::Core ie;

    std::vector<std::string> properties;
    ASSERT_NO_THROW(properties = ie.get_property(target_device, ov::available_devices));

    std::cout << "AVAILABLE_DEVICES: ";
    for (const auto& prop : properties) {
        std::cout << prop << " ";
    }
    std::cout << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::available_devices);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricRangeForAsyncInferRequestsAndPrintNoThrow) {
    ov::Core ie;

    std::tuple<unsigned int, unsigned int, unsigned int> property;
    ASSERT_NO_THROW(property = ie.get_property(target_device, ov::range_for_async_infer_requests));

    std::cout << "RANGE_FOR_ASYNC_INFER_REQUESTS: " << std::get<0>(property) << " " << std::get<1>(property) << " "
              << std::get<2>(property) << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_async_infer_requests);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricRangeForStreamsAndPrintNoThrow) {
    ov::Core ie;

    std::tuple<unsigned int, unsigned int> property;
    ASSERT_NO_THROW(property = ie.get_property(target_device, ov::range_for_streams));

    std::cout << "RANGE_FOR_STREAMS: " << std::get<0>(property) << " " << std::get<1>(property) << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_streams);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricOptimalBatchSizeAndPrintNoThrow) {
    ov::Core ie;

    unsigned int property = 0;
    ASSERT_NO_THROW(property = ie.get_property(target_device, ov::optimal_batch_size));

    std::cout << "OPTIMAL_BATCH_SIZE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::optimal_batch_size);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricFullNameAndPrintNoThrow) {
    ov::Core ie;

    std::string property;
    ASSERT_NO_THROW(property = ie.get_property(target_device, ov::device::full_name));

    std::cout << "FULL_DEVICE_NAME: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::full_name);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricTypeAndPrintNoThrow) {
    ov::Core ie;

    ov::device::Type property = ov::device::Type::INTEGRATED;
    ASSERT_NO_THROW(property = ie.get_property(target_device, ov::device::type));

    std::cout << "DEVICE_TYPE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::type);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricGopsAndPrintNoThrow) {
    ov::Core ie;

    std::map<ov::element::Type, float> properties;
    ASSERT_NO_THROW(properties = ie.get_property(target_device, ov::device::gops));

    std::cout << "DEVICE_GOPS: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << "- " << prop.first << ": " << prop.second << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::gops);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricCapabilitiesAndPrintNoThrow) {
    ov::Core ie;

    std::vector<std::string> properties;
    ASSERT_NO_THROW(properties = ie.get_property(target_device, ov::device::capabilities));

    std::cout << "OPTIMIZATION_CAPABILITIES: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << "- " << prop << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::capabilities);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricDeviceTotalMemSizeAndPrintNoThrow) {
    ov::Core ie;

    uint64_t property = 0;
    ASSERT_NO_THROW(property = ie.get_property(target_device, ov::intel_gpu::device_total_mem_size));

    std::cout << "GPU_DEVICE_TOTAL_MEM_SIZE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::device_total_mem_size);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricUarchVersionAndPrintNoThrow) {
    ov::Core ie;

    std::string property;
    ASSERT_NO_THROW(property = ie.get_property(target_device, ov::intel_gpu::uarch_version));

    std::cout << "GPU_UARCH_VERSION: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::uarch_version);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricExecutionUnitsCountAndPrintNoThrow) {
    ov::Core ie;

    int32_t property = 0;
    ASSERT_NO_THROW(property = ie.get_property(target_device, ov::intel_gpu::execution_units_count));

    std::cout << "GPU_EXECUTION_UNITS_COUNT: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::execution_units_count);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricMemoryStatisticsAndPrintNoThrow) {
    ov::Core ie;

    std::map<std::string, uint64_t> properties;
    ASSERT_NO_THROW(properties = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

    std::cout << "GPU_MEMORY_STATISTICS: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << " " << prop.first << " - " << prop.second << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::memory_statistics);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetPerformanceModeNoThrow) {
    ov::Core ie;

    ov::hint::PerformanceMode defaultMode{};
    ASSERT_NO_THROW(defaultMode = ie.get_property(target_device, ov::hint::performance_mode));

    std::cout << "Default PERFORMANCE_HINT: \"" << defaultMode << "\"" << std::endl;

    ie.set_property(target_device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    ASSERT_EQ(ov::hint::PerformanceMode::LATENCY, ie.get_property(target_device, ov::hint::performance_mode));
    ie.set_property(target_device, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    ASSERT_EQ(ov::hint::PerformanceMode::THROUGHPUT, ie.get_property(target_device, ov::hint::performance_mode));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::hint::performance_mode);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetEnableProfilingNoThrow) {
    ov::Core ie;

    bool defaultValue = false;
    ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, ov::enable_profiling));

    std::cout << "Default PERF_COUNT: " << defaultValue << std::endl;

    ie.set_property(target_device, ov::enable_profiling(true));
    ASSERT_EQ(true, ie.get_property(target_device, ov::enable_profiling));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::enable_profiling);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetInferencePrecisionNoThrow) {
    ov::Core ie;
    auto value = ov::element::undefined;
    const auto expected_default_precision = ov::element::f16;

    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::hint::inference_precision));
    ASSERT_EQ(expected_default_precision, value);

    const auto forced_precision = ov::element::f32;

    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::hint::inference_precision(forced_precision)));
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::hint::inference_precision));
    ASSERT_EQ(value, forced_precision);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto forced_precision_deprecated = ov::element::f16;
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::hint::inference_precision(forced_precision_deprecated)));
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::hint::inference_precision));
    ASSERT_EQ(value, forced_precision_deprecated);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetModelPriorityNoThrow) {
    ov::Core ie;

    ov::hint::Priority defaultValue{};
    ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, ov::hint::model_priority));

    std::cout << "Default model_priority: " << defaultValue << std::endl;

    ie.set_property(target_device, ov::hint::model_priority(ov::hint::Priority::HIGH));
    ASSERT_EQ(ov::hint::Priority::HIGH, ie.get_property(target_device, ov::hint::model_priority));
    ASSERT_EQ(ov::hint::Priority::MEDIUM, ie.get_property(target_device, ov::intel_gpu::hint::queue_priority));
    ie.set_property(target_device, ov::hint::model_priority(ov::hint::Priority::LOW));
    ASSERT_EQ(ov::hint::Priority::LOW, ie.get_property(target_device, ov::hint::model_priority));
    ASSERT_EQ(ov::hint::Priority::MEDIUM, ie.get_property(target_device, ov::intel_gpu::hint::queue_priority));
    ie.set_property(target_device, ov::hint::model_priority(ov::hint::Priority::MEDIUM));
    ASSERT_EQ(ov::hint::Priority::MEDIUM, ie.get_property(target_device, ov::hint::model_priority));
    ASSERT_EQ(ov::hint::Priority::MEDIUM, ie.get_property(target_device, ov::intel_gpu::hint::queue_priority));
    ie.set_property(target_device, ov::intel_gpu::hint::queue_priority(ov::hint::Priority::HIGH));
    ASSERT_EQ(ov::hint::Priority::HIGH, ie.get_property(target_device, ov::intel_gpu::hint::queue_priority));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::hint::model_priority);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetQueuePriorityNoThrow) {
    ov::Core ie;

    ov::hint::Priority defaultValue{};
    ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, ov::intel_gpu::hint::queue_priority));

    std::cout << "Default GPU_QUEUE_PRIORITY: " << defaultValue << std::endl;

    ie.set_property(target_device, ov::intel_gpu::hint::queue_priority(ov::hint::Priority::HIGH));
    ASSERT_EQ(ov::hint::Priority::HIGH, ie.get_property(target_device, ov::intel_gpu::hint::queue_priority));
    ie.set_property(target_device, ov::intel_gpu::hint::queue_priority(ov::hint::Priority::LOW));
    ASSERT_EQ(ov::hint::Priority::LOW, ie.get_property(target_device, ov::intel_gpu::hint::queue_priority));
    ie.set_property(target_device, ov::intel_gpu::hint::queue_priority(ov::hint::Priority::MEDIUM));
    ASSERT_EQ(ov::hint::Priority::MEDIUM, ie.get_property(target_device, ov::intel_gpu::hint::queue_priority));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::hint::queue_priority);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetThrottleLevelNoThrow) {
    ov::Core ie;

    ov::intel_gpu::hint::ThrottleLevel defaultValue{};
    ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, ov::intel_gpu::hint::queue_throttle));

    std::cout << "Default GPU_QUEUE_THROTTLE: " << defaultValue << std::endl;

    ie.set_property(target_device, ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::HIGH));
    ASSERT_EQ(ov::intel_gpu::hint::ThrottleLevel::HIGH,
              ie.get_property(target_device, ov::intel_gpu::hint::queue_throttle));
    ie.set_property(target_device, ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::LOW));
    ASSERT_EQ(ov::intel_gpu::hint::ThrottleLevel::LOW,
              ie.get_property(target_device, ov::intel_gpu::hint::queue_throttle));
    ie.set_property(target_device, ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::MEDIUM));
    ASSERT_EQ(ov::intel_gpu::hint::ThrottleLevel::MEDIUM,
              ie.get_property(target_device, ov::intel_gpu::hint::queue_throttle));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::hint::queue_throttle);
}

TEST_P(OVClassGetPropertyTest_GPU, CanSetDefaultValueBackToPluginNewAPI) {
    ov::Core ie;

    std::vector<ov::PropertyName> properties;
    ASSERT_NO_THROW(properties = ie.get_property(target_device, ov::supported_properties));

    std::cout << "SUPPORTED_PROPERTIES:" << std::endl;
    for (const auto& property : properties) {
        ov::Any prop;
        if (property.is_mutable()) {
            std::cout << "RW: " << property << " ";
            ASSERT_NO_THROW(prop = ie.get_property(target_device, property));
            prop.print(std::cout);
            std::cout << std::endl;
            ASSERT_NO_THROW(ie.set_property(target_device, {{property, prop}}));
        } else {
            std::cout << "RO: " << property << " ";
            ASSERT_NO_THROW(prop = ie.get_property(target_device, property));
            prop.print(std::cout);
            std::cout << std::endl;
        }
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

INSTANTIATE_TEST_SUITE_P(nightly_OVGetMetricPropsTest, OVClassGetPropertyTest_GPU, ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_OPTIMAL_BATCH_SIZE = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_OPTIMAL_BATCH_SIZE, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    unsigned int p = 0;

    ov::AnyMap _options = {ov::hint::model(simpleNetwork)};
    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::optimal_batch_size.name(), _options).as<unsigned int>());

    std::cout << "GPU device optimal batch size: " << p << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::optimal_batch_size);
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCompiledModelGetPropertyTest,
                         OVGetMetricPropsTest_GPU_OPTIMAL_BATCH_SIZE,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_MAX_BATCH_SIZE_DEFAULT = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_MAX_BATCH_SIZE_DEFAULT, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    unsigned int p = 0;

    ov::AnyMap _options = {ov::hint::model(simpleNetwork)};
    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::max_batch_size.name(), _options).as<unsigned int>());

    std::cout << "GPU device max available batch size: " << p << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::max_batch_size);
}

INSTANTIATE_TEST_SUITE_P(nightly_IEClassExecutableNetworkGetMetricTest,
                         OVGetMetricPropsTest_GPU_MAX_BATCH_SIZE_DEFAULT,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    unsigned int p = 0;
    auto exec_net1 = ie.compile_model(simpleNetwork, target_device);

    uint32_t n_streams = 2;
    int64_t available_device_mem_size = 1073741824;
    ov::AnyMap _options = {ov::hint::model(simpleNetwork),
                           ov::num_streams(n_streams),
                           ov::intel_gpu::hint::available_device_mem(available_device_mem_size)};

    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::max_batch_size.name(), _options).as<unsigned int>());

    std::cout << "GPU device max available batch size: " << p << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::max_batch_size);
}

INSTANTIATE_TEST_SUITE_P(nightly_IEClassExecutableNetworkGetMetricTest,
                         OVGetMetricPropsTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_DEFAULT = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_DEFAULT, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    std::map<std::string, uint64_t> p;

    auto exec_net = ie.compile_model(simpleNetwork, target_device);

    ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

    ASSERT_FALSE(p.empty());
    std::cout << "Memory Statistics: " << std::endl;
    for (auto&& kv : p) {
        ASSERT_NE(kv.second, 0);
        std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::memory_statistics);
}

INSTANTIATE_TEST_SUITE_P(nightly_IEClassGetMetricTest,
                         OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_DEFAULT,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    std::map<std::string, uint64_t> t1;
    std::map<std::string, uint64_t> t2;

    auto exec_net1 = ie.compile_model(simpleNetwork, target_device);

    ASSERT_NO_THROW(t1 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

    ASSERT_FALSE(t1.empty());
    for (auto&& kv : t1) {
        ASSERT_NE(kv.second, 0);
    }

    auto exec_net2 = ie.compile_model(simpleNetwork, target_device);

    ASSERT_NO_THROW(t2 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

    ASSERT_FALSE(t2.empty());
    for (auto&& kv : t2) {
        ASSERT_NE(kv.second, 0);
        auto iter = t1.find(kv.first);
        if (iter != t1.end()) {
            ASSERT_EQ(kv.second, t1[kv.first] * 2);
        }
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::memory_statistics);
}

INSTANTIATE_TEST_SUITE_P(nightly_IEClassGetMetricTest,
                         OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_CHECK_VALUES = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_CHECK_VALUES, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    std::map<std::string, uint64_t> t1;

    ASSERT_NO_THROW(t1 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));
    ASSERT_TRUE(t1.empty());

    {
        auto exec_net1 = ie.compile_model(simpleNetwork, target_device);

        std::map<std::string, uint64_t> t2;
        ASSERT_NO_THROW(t2 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

        ASSERT_FALSE(t2.empty());
        for (auto&& kv : t2) {
            ASSERT_NE(kv.second, 0);
        }
        {
            auto exec_net2 = ie.compile_model(actualNetwork, target_device);

            std::map<std::string, uint64_t> t3;
            ASSERT_NO_THROW(t3 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

            ASSERT_FALSE(t3.empty());
            for (auto&& kv : t3) {
                ASSERT_NE(kv.second, 0);
            }
        }
        std::map<std::string, uint64_t> t4;
        ASSERT_NO_THROW(t4 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

        ASSERT_FALSE(t4.empty());
        for (auto&& kv : t4) {
            ASSERT_NE(kv.second, 0);
            if (kv.first.find("_cur") != std::string::npos) {
                auto iter = t2.find(kv.first);
                if (iter != t2.end()) {
                    ASSERT_EQ(t2[kv.first], kv.second);
                }
            }
        }
    }
    std::map<std::string, uint64_t> t5;
    ASSERT_NO_THROW(t5 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

    ASSERT_FALSE(t5.empty());
    for (auto&& kv : t5) {
        if (kv.first.find("_cur") != std::string::npos) {
            ASSERT_EQ(kv.second, 0);
        }
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::memory_statistics);
}

INSTANTIATE_TEST_SUITE_P(nightly_IEClassGetMetricTest,
                         OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_CHECK_VALUES,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_MULTI_THREADS = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_MULTI_THREADS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    std::map<std::string, uint64_t> t1;
    std::map<std::string, uint64_t> t2;

    std::atomic<uint32_t> counter{0u};
    std::vector<std::thread> threads(2);
    // key: thread id, value: executable network
    std::map<uint32_t, ov::CompiledModel> exec_net_map;
    std::vector<std::shared_ptr<ngraph::Function>> networks;
    networks.emplace_back(simpleNetwork);
    networks.emplace_back(simpleNetwork);

    auto exec_net1 = ie.compile_model(simpleNetwork, target_device);

    ASSERT_NO_THROW(t1 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

    ASSERT_FALSE(t1.empty());
    for (auto&& kv : t1) {
        ASSERT_NE(kv.second, 0);
    }

    for (auto& thread : threads) {
        thread = std::thread([&]() {
            auto value = counter++;
            exec_net_map[value] = ie.compile_model(networks[value], target_device);
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    ASSERT_NO_THROW(t2 = ie.get_property(target_device, ov::intel_gpu::memory_statistics));

    ASSERT_FALSE(t2.empty());
    for (auto&& kv : t2) {
        ASSERT_NE(kv.second, 0);
        auto iter = t1.find(kv.first);
        if (iter != t1.end()) {
            ASSERT_EQ(kv.second, t1[kv.first] * 3);
        }
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_gpu::memory_statistics);
}

INSTANTIATE_TEST_SUITE_P(nightly_IEClassGetMetricTest,
                         OVGetMetricPropsTest_GPU_MEMORY_STATISTICS_MULTI_THREADS,
                         ::testing::Values("GPU"));

using OVGetMetricPropsTest_CACHING_PROPERTIES = OVClassBaseTestP;
TEST_P(OVGetMetricPropsTest_CACHING_PROPERTIES, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> caching_properties = {};
    const std::vector<ov::PropertyName> expected_properties = {
        ov::device::architecture.name(),
        ov::intel_gpu::execution_units_count.name(),
        ov::intel_gpu::driver_version.name(),
        ov::hint::inference_precision.name(),
        ov::hint::execution_mode.name(),
    };

    ASSERT_NO_THROW(caching_properties = ie.get_property(target_device, ov::internal::caching_properties));

    std::cout << "GPU Caching properties: " << std::endl;
    for (auto& prop : caching_properties) {
        std::cout << prop << std::endl;
    }

    ASSERT_EQ(caching_properties.size(), expected_properties.size());

    for (const auto& property_name : expected_properties) {
        ASSERT_TRUE(std::find(caching_properties.begin(), caching_properties.end(), property_name) !=
                    caching_properties.end());
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::internal::caching_properties);
}

INSTANTIATE_TEST_SUITE_P(nightly_OVGetMetricPropsTest,
                         OVGetMetricPropsTest_CACHING_PROPERTIES,
                         ::testing::Values("GPU"));

// GetConfig / SetConfig for specific device

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSpecificDevice0Test, OVSpecificDeviceSetConfigTest, ::testing::Values("GPU.0"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSpecificDevice1Test, OVSpecificDeviceSetConfigTest, ::testing::Values("GPU.1"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSpecificDevice0Test, OVSpecificDeviceGetConfigTest, ::testing::Values("GPU.0"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSpecificDevice1Test, OVSpecificDeviceGetConfigTest, ::testing::Values("GPU.1"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSpecificDevice0Test, OVSpecificDeviceTestSetConfig, ::testing::Values("GPU.0"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSpecificDevice1Test, OVSpecificDeviceTestSetConfig, ::testing::Values("GPU.1"));

// Set default device ID

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSetDefaultDeviceIDPropTest,
                         OVClassSetDefaultDeviceIDPropTest,
                         ::testing::Values(std::make_pair("GPU", "1")));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(nightly_OVPropertiesDefaultSupportedTests,
                         OVPropertiesDefaultSupportedTests,
                         ::testing::Values("GPU"));

}  // namespace
