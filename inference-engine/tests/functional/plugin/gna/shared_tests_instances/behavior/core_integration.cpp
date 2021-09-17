// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/core_integration.hpp"
#include <gna/gna_config.hpp>

using namespace BehaviorTestsDefinitions;

namespace {

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassBasicTestP, IEClassBasicTestP,
        ::testing::Values(std::make_pair("GNAPlugin", "GNA")));

// TODO
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values("GNA"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GNA", "MULTI", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GNA", "MULTI", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GNA", "MULTI", "HETERO"));

// TODO: Issue: 30198
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("GNA"));

// TODO: Issue: 30199
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("GNA", "MULTI", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GNA", "MULTI", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values("GNA"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values("GNA"));

//
// Executable Network GetMetric
//

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GNA" /*, "MULTI:GNA", "HETERO:GNA" */));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GNA" /*, "MULTI:GNA",  "HETERO:GNA" */));

// TODO: this metric is not supported by the plugin
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GNA", "MULTI:GNA", "HETERO:GNA"));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("GNA"/*, "MULTI:GNA", "HETERO:GNA" */));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::Values("GNA", /* "MULTI:GNA", */ "HETERO:GNA"));

//
// Executable Network GetConfig / SetConfig
//

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values("GNA"));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values("GNA"));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkSupportedConfigTest, IEClassExecutableNetworkSupportedConfigTest,
        ::testing::Combine(::testing::Values("GNA"),
                           ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_HW),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_EXACT),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_AUTO))));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkUnsupportedConfigTest, IEClassExecutableNetworkUnsupportedConfigTest,
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

    ASSERT_THROW(exeNetwork.SetConfig({{configKey, configValue}}), Exception);
}

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkSetConfigFromFp32Test, IEClassExecutableNetworkSetConfigFromFp32Test,
        ::testing::Combine(::testing::Values("GNA"),
                           ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_HW),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_EXACT),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_SW_FP32),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_AUTO))));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values("GNA"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values("GNA"));

//
// Hetero Executable Network GetMetric
//

// TODO: verify hetero interop
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GNA"));

// TODO: verify hetero interop
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GNA"));

// TODO: verify hetero interop
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values("GNA"));
} // namespace
