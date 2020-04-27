// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_class.hpp"

#ifdef _WIN32
# include "gpu/gpu_context_api_dx.hpp"
#elif defined ENABLE_LIBVA
# include <gpu/gpu_context_api_va.hpp>
#endif
#include "gpu/gpu_context_api_ocl.hpp"

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassCommon, IEClassBasicTestP,
        ::testing::Values(std::make_pair("clDNNPlugin", "GPU")));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values("GPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values("GPU"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values("GPU"));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU"));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values("GPU"));

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values("GPU"));

// IE Class Query network

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values("GPU"));

// IE Class Load network

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassLoadNetworkAfterCoreRecreateTest,
        ::testing::Values("GPU"));
