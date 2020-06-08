// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_class.hpp"
#include <gna/gna_config.hpp>

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassBasicTestP, IEClassBasicTestP,
        ::testing::Values(std::make_pair("GNAPlugin", "GNA")));

// TODO
INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values("GNA"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GNA", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GNA", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GNA"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GNA", "MULTI", "HETERO"));

// TODO: Issue: 30198
INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("GNA"));

// TODO: Issue: 30199
INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("GNA"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("GNA", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GNA", "MULTI", "HETERO"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values("GNA"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values("GNA"));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
    ::testing::Values("GNA" /*, "MULTI:GNA", "HETERO:GNA" */));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
    ::testing::Values("GNA" /*, "MULTI:GNA",  "HETERO:GNA" */));

// TODO: this metric is not supported by the plugin
INSTANTIATE_TEST_CASE_P(
   DISABLED_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
   ::testing::Values("GNA", "MULTI:GNA", "HETERO:GNA"));

INSTANTIATE_TEST_CASE_P(
   IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
   ::testing::Values("GNA"/*, "MULTI:GNA", "HETERO:GNA" */));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
    ::testing::Values("GNA", /* "MULTI:GNA", */ "HETERO:GNA"));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
    ::testing::Values("GNA"));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
    ::testing::Values("GNA"));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkSupportedConfigTest, IEClassExecutableNetworkSupportedConfigTest,
    ::testing::Combine(::testing::Values("GNA"),
                       ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_HW),
                                         std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW),
                                         std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_EXACT),
                                         std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_AUTO))));

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkUnsupportedConfigTest, IEClassExecutableNetworkUnsupportedConfigTest,
    ::testing::Combine(::testing::Values("GNA"),
                       ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_FP32),
                                         std::make_pair(GNA_CONFIG_KEY(SCALE_FACTOR), "5"),
                                         std::make_pair(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)),
                                         std::make_pair(GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)))));

using IEClassExecutableNetworkSetConfigFromFp32Test = IEClassExecutableNetworkGetMetricTestForSpecificConfig;
TEST_P(IEClassExecutableNetworkSetConfigFromFp32Test, SetConfigFromFp32Throws) {
    Core ie;

    std::map<std::string, std::string> initialConfig;
    initialConfig[GNA_CONFIG_KEY(DEVICE_MODE)] = GNAConfigParams::GNA_SW_FP32;
    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleNetwork, deviceName, initialConfig);

    ASSERT_THROW(exeNetwork.SetConfig({ { configKey, configValue } }), InferenceEngineException);
}

INSTANTIATE_TEST_CASE_P(
    IEClassExecutableNetworkSetConfigFromFp32Test, IEClassExecutableNetworkSetConfigFromFp32Test,
    ::testing::Combine(::testing::Values("GNA"),
                       ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_HW),
                                         std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW),
                                         std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_EXACT),
                                         std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_FP32),
                                         std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_AUTO))));

// IE Class Query network

INSTANTIATE_TEST_CASE_P(
    IEClassQueryNetworkTest, IEClassQueryNetworkTest,
    ::testing::Values("GNA"));

// IE Class Load network

INSTANTIATE_TEST_CASE_P(
   IEClassLoadNetworkTest, IEClassLoadNetworkTest,
   ::testing::Values("GNA"));

//
// Hetero Executable Network GetMetric
//

// TODO: verify hetero interop
INSTANTIATE_TEST_CASE_P(
   DISABLED_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
   ::testing::Values("GNA"));

// TODO: verify hetero interop
INSTANTIATE_TEST_CASE_P(
   DISABLED_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
   ::testing::Values("GNA"));

// TODO: verify hetero interop
INSTANTIATE_TEST_CASE_P(
   DISABLED_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
   ::testing::Values("GNA"));

INSTANTIATE_TEST_CASE_P(
   IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
   ::testing::Values("GNA"));
