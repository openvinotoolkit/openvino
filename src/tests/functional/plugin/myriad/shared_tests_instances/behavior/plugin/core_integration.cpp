// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/skip_tests_config.hpp>
#include <openvino/runtime/properties.hpp>
#include "behavior/plugin/core_integration.hpp"
#include "common_test_utils/file_utils.hpp"

using namespace BehaviorTestsDefinitions;
using IEClassGetMetricTest_nightly = IEClassGetMetricTest;
using IEClassGetConfigTest_nightly = IEClassGetConfigTest;

namespace {
std::vector<std::string> devices = {
    std::string(CommonTestUtils::DEVICE_MYRIAD),
};

std::pair<std::string, std::string> plugins[] = {
        std::make_pair(std::string("openvino_intel_myriad_plugin"), std::string(CommonTestUtils::DEVICE_MYRIAD)),
};

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        IEClassBasicTestP_smoke, IEClassBasicTestP,
        ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_SUITE_P(
        IEClassNetworkTestP_smoke, IEClassNetworkTestP,
        ::testing::ValuesIn(devices));

//
// IEClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

using IEClassNetworkTestP_VPU_GetMetric = IEClassNetworkTestP;

TEST_P(IEClassNetworkTestP_VPU_GetMetric, smoke_OptimizationCapabilitiesReturnsFP16) {
    InferenceEngine::Core ie;
    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(OPTIMIZATION_CAPABILITIES))

    InferenceEngine::Parameter optimizationCapabilitiesParameter;
    ASSERT_NO_THROW(optimizationCapabilitiesParameter = ie.GetMetric(deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES)));

    const auto optimizationCapabilities = optimizationCapabilitiesParameter.as<std::vector<std::string>>();
    ASSERT_EQ(optimizationCapabilities.size(), 2);
    ASSERT_NE(std::find(optimizationCapabilities.begin(),
                        optimizationCapabilities.end(),
                        ov::device::capability::EXPORT_IMPORT),
              optimizationCapabilities.end());
    ASSERT_NE(std::find(optimizationCapabilities.begin(), optimizationCapabilities.end(), ov::device::capability::FP16),
              optimizationCapabilities.end());
}

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricP, IEClassNetworkTestP_VPU_GetMetric,
        ::testing::ValuesIn(devices));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_ThrowUnsupported,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassGetMetricTest_nightly,
        IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::ValuesIn(devices));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        IEClassGetConfigTest_nightly,
        IEClassGetConfigTest,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassGetConfigTest_nightly,
        IEClassGetConfigTest_ThrowUnsupported,
        ::testing::ValuesIn(devices));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        DISABLED_IEClassQueryNetworkTest_smoke,
        IEClassQueryNetworkTest,
        ::testing::ValuesIn(devices));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        IEClassLoadNetworkTest_smoke,
        IEClassLoadNetworkTest,
        ::testing::ValuesIn(devices));
} // namespace