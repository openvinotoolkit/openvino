// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/get_metric.hpp"
#include "conformance.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine::PluginConfigParams;

namespace {

std::string generateComplexDeviceName(const std::string& deviceName) {
    return deviceName + ":" + ConformanceTests::targetDevice;
}

std::vector<std::string> returnAllPosibleDeviceCombination() {
    std::vector<std::string> res{ConformanceTests::targetDevice};
    std::vector<std::string> devices{CommonTestUtils::DEVICE_HETERO, CommonTestUtils::DEVICE_AUTO, CommonTestUtils::DEVICE_MULTI};
    for (const auto& device : devices) {
        res.emplace_back(generateComplexDeviceName(device));
    }
    return res;
}

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassImportExportTestP, IEClassImportExportTestP,
        ::testing::Values(generateComplexDeviceName(CommonTestUtils::DEVICE_HETERO)));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(returnAllPosibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(returnAllPosibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(returnAllPosibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(returnAllPosibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(returnAllPosibleDeviceCombination()));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values(ConformanceTests::targetDevice));

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values(ConformanceTests::targetDevice));

} // namespace