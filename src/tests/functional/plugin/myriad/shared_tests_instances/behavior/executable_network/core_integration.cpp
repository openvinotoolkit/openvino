// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/skip_tests_config.hpp>
#include "behavior/executable_network/get_metric.hpp"
#include "common_test_utils/file_utils.hpp"

using namespace BehaviorTestsDefinitions;
using IEClassExecutableNetworkGetMetricTest_nightly = IEClassExecutableNetworkGetMetricTest;
using IEClassExecutableNetworkGetConfigTest_nightly = IEClassExecutableNetworkGetConfigTest;

namespace {
std::vector<std::string> devices = {
    std::string(CommonTestUtils::DEVICE_MYRIAD),
};

std::pair<std::string, std::string> plugins[] = {
        std::make_pair(std::string("openvino_intel_myriad_plugin"), std::string(CommonTestUtils::DEVICE_MYRIAD)),
};

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassImportExportTestP, IEClassImportExportTestP,
        ::testing::Values(std::string(CommonTestUtils::DEVICE_MYRIAD), "HETERO:" + std::string(CommonTestUtils::DEVICE_MYRIAD)));

#if defined(ENABLE_INTEL_CPU) && ENABLE_INTEL_CPU

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassImportExportTestP_HETERO_CPU, IEClassImportExportTestP,
        ::testing::Values("HETERO:" + std::string(CommonTestUtils::DEVICE_MYRIAD) + ",CPU"));
#endif

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        IEClassExecutableNetworkGetMetricTest_nightly,
        IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(devices));

//
// Executable Network GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        IEClassExecutableNetworkGetConfigTest_nightly,
        IEClassExecutableNetworkGetConfigTest,
        ::testing::ValuesIn(devices));
} // namespace