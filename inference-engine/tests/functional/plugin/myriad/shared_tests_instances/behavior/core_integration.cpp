// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/skip_tests_config.hpp>
#include "behavior/core_integration.hpp"
#include "common_test_utils/file_utils.hpp"

using namespace BehaviorTestsDefinitions;
using IEClassExecutableNetworkGetMetricTest_nightly = IEClassExecutableNetworkGetMetricTest;
using IEClassExecutableNetworkGetConfigTest_nightly = IEClassExecutableNetworkGetConfigTest;

using IEClassGetMetricTest_nightly = IEClassGetMetricTest;
using IEClassGetConfigTest_nightly = IEClassGetConfigTest;

namespace {
std::vector<std::string> devices = {
    std::string(CommonTestUtils::DEVICE_MYRIAD),
};

std::pair<std::string, std::string> plugins[] = {
        std::make_pair(std::string("myriadPlugin"), std::string(CommonTestUtils::DEVICE_MYRIAD)),
};

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_CASE_P(
        IEClassBasicTestP_smoke, IEClassBasicTestP,
        ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_CASE_P(
        IEClassNetworkTestP_smoke, IEClassNetworkTestP,
        ::testing::ValuesIn(devices));

//
// IEClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

using IEClassNetworkTestP_VPU_GetMetric = IEClassNetworkTestP;

TEST_P(IEClassNetworkTestP_VPU_GetMetric, smoke_OptimizationCapabilitiesReturnsFP16) {
    Core ie;
    ASSERT_METRIC_SUPPORTED(METRIC_KEY(OPTIMIZATION_CAPABILITIES))

    Parameter optimizationCapabilitiesParameter;
    ASSERT_NO_THROW(optimizationCapabilitiesParameter = ie.GetMetric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));

    const auto optimizationCapabilities = optimizationCapabilitiesParameter.as<std::vector<std::string>>();
    ASSERT_EQ(optimizationCapabilities.size(), 1);
    ASSERT_EQ(optimizationCapabilities.front(), METRIC_VALUE(FP16));
}

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassGetMetricP, IEClassNetworkTestP_VPU_GetMetric,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassImportExportTestP, IEClassImportExportTestP,
        ::testing::Values(std::string(CommonTestUtils::DEVICE_MYRIAD), "HETERO:" + std::string(CommonTestUtils::DEVICE_MYRIAD)));

#if defined(ENABLE_MKL_DNN) && ENABLE_MKL_DNN

INSTANTIATE_TEST_CASE_P(
        smoke_IEClassImportExportTestP_HETERO_CPU, IEClassImportExportTestP,
        ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_MYRIAD) + ",CPU"));
#endif

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(devices));

//
// Executable Network GetConfig
//

INSTANTIATE_TEST_CASE_P(
        IEClassExecutableNetworkGetConfigTest_nightly,
        IEClassExecutableNetworkGetConfigTest,
        ::testing::ValuesIn(devices));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_ThrowUnsupported,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::ValuesIn(devices));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_CASE_P(
        IEClassGetConfigTest_nightly,
        IEClassGetConfigTest,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_CASE_P(
        IEClassGetConfigTest_nightly,
        IEClassGetConfigTest_ThrowUnsupported,
        ::testing::ValuesIn(devices));

// IE Class Query network

INSTANTIATE_TEST_CASE_P(
        DISABLED_IEClassQueryNetworkTest_smoke,
        IEClassQueryNetworkTest,
        ::testing::ValuesIn(devices));

// IE Class Load network

INSTANTIATE_TEST_CASE_P(
        IEClassLoadNetworkTest_smoke,
        IEClassLoadNetworkTest,
        ::testing::ValuesIn(devices));
} // namespace