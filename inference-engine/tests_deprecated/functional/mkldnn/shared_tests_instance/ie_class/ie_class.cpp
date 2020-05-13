// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_class.hpp"

using namespace InferenceEngine::PluginConfigParams;

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassCommon, IEClassBasicTestP,
        ::testing::Values(std::make_pair("MKLDNNPlugin", "CPU")));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassImportExportTestP, IEClassImportExportTestP,
        ::testing::Values("HETERO:CPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("CPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("CPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("CPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("CPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("CPU", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values("CPU"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values("CPU"));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU"));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values("CPU"));

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values("CPU"));

//////////////////////////////////////////////////////////////////////////////////////////

TEST_F(IEClassBasicTest, smoke_SetConfigAfterCreatedThrow) {
    Core ie;
    std::string value = { };

    ASSERT_NO_THROW(ie.SetConfig({ { KEY_CPU_THREADS_NUM, "1" } }, "CPU"));
    ASSERT_NO_THROW(value = ie.GetConfig("CPU", KEY_CPU_THREADS_NUM).as<std::string>());
    ASSERT_EQ("1", value);

    ASSERT_NO_THROW(ie.SetConfig({ { KEY_CPU_THREADS_NUM, "4" } }, "CPU"));
    ASSERT_NO_THROW(value = ie.GetConfig("CPU", KEY_CPU_THREADS_NUM).as<std::string>());
    ASSERT_EQ("4", value);
}

// IE Class Query network

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values("CPU"));

// IE Class Load network

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values("CPU"));
