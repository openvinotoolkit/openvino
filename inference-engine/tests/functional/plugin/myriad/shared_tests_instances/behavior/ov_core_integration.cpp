// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_core_integration.hpp"

#include <functional_test_utils/skip_tests_config.hpp>

#include "common_test_utils/file_utils.hpp"

using namespace ov::test;

using OVClassExecutableNetworkGetMetricTest_nightly = OVClassExecutableNetworkGetMetricTest;
using OVClassExecutableNetworkGetConfigTest_nightly = OVClassExecutableNetworkGetConfigTest;

using OVClassGetMetricTest_nightly = OVClassGetMetricTest;
using OVClassGetConfigTest_nightly = OVClassGetConfigTest;

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

INSTANTIATE_TEST_SUITE_P(OVClassBasicTestP_smoke, OVClassBasicTestP, ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_SUITE_P(OVClassNetworkTestP_smoke, OVClassNetworkTestP, ::testing::ValuesIn(devices));

//
// OVClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

using OVClassNetworkTestP_VPU_GetMetric = OVClassNetworkTestP;

TEST_P(OVClassNetworkTestP_VPU_GetMetric, smoke_OptimizationCapabilitiesReturnsFP16) {
    ov::runtime::Core ie;
    ASSERT_METRIC_SUPPORTED(METRIC_KEY(OPTIMIZATION_CAPABILITIES))

    ov::runtime::Parameter optimizationCapabilitiesParameter;
    ASSERT_NO_THROW(optimizationCapabilitiesParameter =
                        ie.get_metric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));

    const auto optimizationCapabilities = optimizationCapabilitiesParameter.as<std::vector<std::string>>();
    ASSERT_EQ(optimizationCapabilities.size(), 1);
    ASSERT_EQ(optimizationCapabilities.front(), METRIC_VALUE(FP16));
}

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricP, OVClassNetworkTestP_VPU_GetMetric, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP,
                         OVClassImportExportTestP,
                         ::testing::Values(std::string(CommonTestUtils::DEVICE_MYRIAD),
                                           "HETERO:" + std::string(CommonTestUtils::DEVICE_MYRIAD)));

#if defined(ENABLE_MKL_DNN) && ENABLE_MKL_DNN

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP_HETERO_CPU,
                         OVClassImportExportTestP,
                         ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_MYRIAD) + ",CPU"));
#endif

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetMetricTest_nightly,
                         OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
                         ::testing::ValuesIn(devices));

//
// Executable Network GetConfig
//

INSTANTIATE_TEST_SUITE_P(OVClassExecutableNetworkGetConfigTest_nightly,
                         OVClassExecutableNetworkGetConfigTest,
                         ::testing::ValuesIn(devices));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_AVAILABLE_DEVICES,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_SUPPORTED_METRICS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetMetricTest_nightly,
                         OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
                         ::testing::ValuesIn(devices));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(OVClassGetConfigTest_nightly, OVClassGetConfigTest, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(OVClassGetConfigTest_nightly,
                         OVClassGetConfigTest_ThrowUnsupported,
                         ::testing::ValuesIn(devices));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(DISABLED_OVClassQueryNetworkTest_smoke, OVClassQueryNetworkTest, ::testing::ValuesIn(devices));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(OVClassLoadNetworkTest_smoke, OVClassLoadNetworkTest, ::testing::ValuesIn(devices));
}  // namespace
