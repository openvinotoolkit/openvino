// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"

#include <gna/gna_config.hpp>

using namespace ov::test::behavior;

namespace {
//
// Executable Network GetMetric
//

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GNA" /*, "MULTI:GNA", "HETERO:GNA" */));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GNA" /*, "MULTI:GNA",  "HETERO:GNA" */));

// TODO: this metric is not supported by the plugin
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GNA", "MULTI:GNA", "HETERO:GNA"));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("GNA" /*, "MULTI:GNA", "HETERO:GNA" */));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::Values("GNA", /* "MULTI:GNA", */ "HETERO:GNA"));

//
// Executable Network GetConfig / SetConfig
//

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetConfigTest,
        OVClassExecutableNetworkGetConfigTest,
        ::testing::Values("GNA"));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkSetConfigTest,
        OVClassExecutableNetworkSetConfigTest,
        ::testing::Values("GNA"));

IE_SUPPRESS_DEPRECATED_START
// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_OVClassExecutableNetworkSupportedConfigTest,
        OVClassExecutableNetworkSupportedConfigTest,
        ::testing::Combine(
        ::testing::Values("GNA"),
        ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_HW),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_EXACT),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_AUTO))));
IE_SUPPRESS_DEPRECATED_END

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_OVClassExecutableNetworkUnsupportedConfigTest,
        OVClassExecutableNetworkUnsupportedConfigTest,
        ::testing::Combine(::testing::Values("GNA"),
                           ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE),
                                                            InferenceEngine::GNAConfigParams::GNA_SW_FP32),
                                             std::make_pair(GNA_CONFIG_KEY(SCALE_FACTOR), "5"),
                                             std::make_pair(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)),
                                             std::make_pair(GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)))));

using OVClassExecutableNetworkSetConfigFromFp32Test = OVClassExecutableNetworkGetMetricTestForSpecificConfig;

TEST_P(OVClassExecutableNetworkSetConfigFromFp32Test, SetConfigFromFp32Throws) {
ov::runtime::Core ie;

std::map<std::string, std::string> initialConfig;
initialConfig[GNA_CONFIG_KEY(DEVICE_MODE)] = InferenceEngine::GNAConfigParams::GNA_SW_FP32;
ov::runtime::CompiledModel exeNetwork = ie.compile_model(simpleNetwork, deviceName, initialConfig);

ASSERT_THROW(exeNetwork.set_config({{configKey, configValue}}), ov::Exception);
}

IE_SUPPRESS_DEPRECATED_START
// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_OVClassExecutableNetworkSetConfigFromFp32Test,
        OVClassExecutableNetworkSetConfigFromFp32Test,
        ::testing::Combine(
        ::testing::Values("GNA"),
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
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassHeteroExecutableNetworlGetMetricTest,
        OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GNA"));

// TODO: verify hetero interop
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassHeteroExecutableNetworlGetMetricTest,
        OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GNA"));

// TODO: verify hetero interop
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassHeteroExecutableNetworlGetMetricTest,
        OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroExecutableNetworlGetMetricTest,
        OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values("GNA"));
}  // namespace
