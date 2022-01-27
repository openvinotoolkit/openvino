// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/get_metric.hpp"
#include <gna/gna_config.hpp>

using namespace BehaviorTestsDefinitions;

namespace {
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

IE_SUPPRESS_DEPRECATED_START
// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkSupportedConfigTest, IEClassExecutableNetworkSupportedConfigTest,
        ::testing::Combine(::testing::Values("GNA"),
                           ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_HW),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_EXACT),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_AUTO))));
IE_SUPPRESS_DEPRECATED_END

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkUnsupportedConfigTest, IEClassExecutableNetworkUnsupportedConfigTest,
        ::testing::Combine(::testing::Values("GNA"),
                           ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_FP32),
                                             std::make_pair(GNA_CONFIG_KEY(SCALE_FACTOR), "5"),
                                             std::make_pair(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)),
                                             std::make_pair(GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)))));

using IEClassExecutableNetworkSetConfigFromFp32Test = IEClassExecutableNetworkGetMetricTestForSpecificConfig;

TEST_P(IEClassExecutableNetworkSetConfigFromFp32Test, SetConfigFromFp32Throws) {
    InferenceEngine::Core ie;

    std::map<std::string, std::string> initialConfig;
    initialConfig[GNA_CONFIG_KEY(DEVICE_MODE)] = InferenceEngine::GNAConfigParams::GNA_SW_FP32;
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName, initialConfig);

    ASSERT_THROW(exeNetwork.SetConfig({{configKey, configValue}}), InferenceEngine::Exception);
}

IE_SUPPRESS_DEPRECATED_START
// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_IEClassExecutableNetworkSetConfigFromFp32Test, IEClassExecutableNetworkSetConfigFromFp32Test,
        ::testing::Combine(::testing::Values("GNA"),
                           ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_HW),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_EXACT),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_FP32),
                                             std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_AUTO))));
IE_SUPPRESS_DEPRECATED_END

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
