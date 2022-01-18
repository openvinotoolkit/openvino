// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"

#ifdef _WIN32
#    include "gpu/gpu_context_api_dx.hpp"
#elif defined ENABLE_LIBVA
#    include <gpu/gpu_context_api_va.hpp>
#endif
#include "gpu/gpu_config.hpp"
#include "gpu/gpu_context_api_ocl.hpp"

using namespace ov::test::behavior;

namespace {
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCommon,
        OVClassBasicTestP,
        ::testing::Values(std::make_pair("ov_intel_gpu_plugin", "GPU")));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassNetworkTestP, OVClassNetworkTestP, ::testing::Values("GPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest, OVClassGetMetricTest_DEVICE_GOPS, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest, OVClassGetMetricTest_DEVICE_TYPE, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetConfigTest,
        OVClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetAvailableDevices, OVClassGetAvailableDevices, ::testing::Values("GPU"));

//
// GPU specific metrics
//
using OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)));
    uint64_t t = p;

    std::cout << "GPU device total memory size: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE,
        ::testing::Values("GPU"));

using OVClassGetMetricTest_GPU_UARCH_VERSION = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_UARCH_VERSION, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, GPU_METRIC_KEY(UARCH_VERSION)));
    std::string t = p;

    std::cout << "GPU device uarch: " << t << std::endl;
    ASSERT_METRIC_SUPPORTED(GPU_METRIC_KEY(UARCH_VERSION));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_UARCH_VERSION,
        ::testing::Values("GPU"));

using OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT, GetMetricAndPrintNoThrow) {
    ov::runtime::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, GPU_METRIC_KEY(EXECUTION_UNITS_COUNT)));
    int t = p;

    std::cout << "GPU EUs count: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED(GPU_METRIC_KEY(EXECUTION_UNITS_COUNT));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT,
        ::testing::Values("GPU"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetConfigTest, OVClassGetConfigTest, ::testing::Values("GPU"));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest, ::testing::Values("GPU"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroExecutableNetworkGetMetricTest,
        OVClassLoadNetworkAfterCoreRecreateTest,
        ::testing::Values("GPU"));

// GetConfig / SetConfig for specific device

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice0Test, OVClassSpecificDeviceTestGetConfig,
        ::testing::Values("GPU.0")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice1Test, OVClassSpecificDeviceTestGetConfig,
        ::testing::Values("GPU.1")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice0Test, OVClassSpecificDeviceTestSetConfig,
        ::testing::Values("GPU.0")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice1Test, OVClassSpecificDeviceTestSetConfig,
        ::testing::Values("GPU.1")
);

// Several devices case

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestLoadNetwork,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestQueryNetwork,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestDefaultCore,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

// Set default device ID

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSetDefaultDeviceIDTest, OVClassSetDefaultDeviceIDTest,
        ::testing::Values(std::make_pair("GPU", "1"))
);

// Set config for all GPU devices

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSetGlobalConfigTest, OVClassSetGlobalConfigTest,
        ::testing::Values("GPU")
);
}  // namespace
