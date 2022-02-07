// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <functional_test_utils/skip_tests_config.hpp>

#include "common_test_utils/file_utils.hpp"

using namespace ov::test::behavior;

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

INSTANTIATE_TEST_SUITE_P(OVClassBasicTestP_smoke, OVClassBasicTestP, ::testing::ValuesIn(plugins));

INSTANTIATE_TEST_SUITE_P(OVClassNetworkTestP_smoke, OVClassNetworkTestP, ::testing::ValuesIn(devices));

//
// OVClassNetworkTestP tests, customized to add SKIP_IF_CURRENT_TEST_IS_DISABLED()
//

using OVClassNetworkTestP_VPU_GetMetric = OVClassNetworkTestP;

TEST_P(OVClassNetworkTestP_VPU_GetMetric, smoke_OptimizationCapabilitiesReturnsFP16) {
    ov::Core ie;
    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::capabilities)
    std::vector<std::string> device_capabilities;
    ASSERT_NO_THROW(device_capabilities = ie.get_property(deviceName, ov::device::capabilities));
    ASSERT_EQ(device_capabilities.size(), 2);
    ASSERT_NE(std::find(device_capabilities.begin(), device_capabilities.end(), ov::device::capability::EXPORT_IMPORT),
              device_capabilities.end());
    ASSERT_NE(std::find(device_capabilities.begin(), device_capabilities.end(), ov::device::capability::FP16),
              device_capabilities.end());
}

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricP, OVClassNetworkTestP_VPU_GetMetric, ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP,
                         OVClassImportExportTestP,
                         ::testing::Values(std::string(CommonTestUtils::DEVICE_MYRIAD),
                                           "HETERO:" + std::string(CommonTestUtils::DEVICE_MYRIAD)));

#if defined(ENABLE_INTEL_CPU) && ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP_HETERO_CPU,
                         OVClassImportExportTestP,
                         ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_MYRIAD) + ",CPU"));
#endif

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