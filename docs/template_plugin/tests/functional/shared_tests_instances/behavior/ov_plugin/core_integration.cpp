// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include <utility>
#include <string>
#include <vector>

#include "behavior/ov_plugin/core_integration.hpp"

using namespace ov::test::behavior;

namespace {

//
// OV Class Common tests with <pluginName, device_name params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassBasicTestP, OVClassBasicTestP,
        ::testing::Values(std::make_pair("openvino_template_plugin", CommonTestUtils::DEVICE_TEMPLATE)));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassNetworkTestP, OVClassNetworkTestP,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_ThrowUnsupported,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest_ThrowUnsupported,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetAvailableDevices, OVClassGetAvailableDevices,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));


//
// OV Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

TEST(OVClassBasicTest, smoke_TEMPLATEGetSetConfigNoThrow) {
    ov::Core core = createCoreWithTemplate();

    auto device_name = CommonTestUtils::DEVICE_TEMPLATE;

    for (auto&& property : core.get_property(device_name, ov::supported_properties)) {
        if (ov::device::id == property) {
            std::cout << ov::device::id.name() << " : " << core.get_property(device_name, ov::device::id) << std::endl;
        } else if (ov::enable_profiling == property) {
            std::cout << ov::enable_profiling.name() << " : " << core.get_property(device_name, ov::enable_profiling) << std::endl;
        } else if (ov::hint::performance_mode == property) {
            std::cout << "Default " << ov::hint::performance_mode.name() << " : " << core.get_property(device_name, ov::hint::performance_mode) << std::endl;
            core.set_property(device_name, ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED));
            ASSERT_EQ(ov::hint::PerformanceMode::UNDEFINED, core.get_property(device_name, ov::hint::performance_mode));
            core.set_property(device_name, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
            ASSERT_EQ(ov::hint::PerformanceMode::THROUGHPUT, core.get_property(device_name, ov::hint::performance_mode));
        }
    }
}

// OV Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

// OV Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));
} // namespace
