// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <string>
#include <vector>

#include "behavior/core_integration.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassBasicTestP, IEClassBasicTestP,
        ::testing::Values(std::make_pair("templatePlugin", "TEMPLATE")));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values("TEMPLATE"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values("TEMPLATE"));


//
// IE Class SetConfig
//

using IEClassSetConfigTestHETERO = IEClassNetworkTest;

TEST_F(IEClassSetConfigTestHETERO, nightly_SetConfigNoThrow) {
    {
        Core ie;
        Parameter p;

        ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), CONFIG_VALUE(YES)}}, "HETERO"));
        ASSERT_NO_THROW(p = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)));
        bool dump = p.as<bool>();

        ASSERT_TRUE(dump);
    }

    {
        Core ie;
        Parameter p;

        ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), CONFIG_VALUE(NO)}}, "HETERO"));
        ASSERT_NO_THROW(p = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)));
        bool dump = p.as<bool>();

        ASSERT_FALSE(dump);
    }

    {
        Core ie;
        Parameter p;

        ASSERT_NO_THROW(ie.GetMetric("HETERO", METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), CONFIG_VALUE(YES)}}, "HETERO"));
        ASSERT_NO_THROW(p = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)));
        bool dump = p.as<bool>();

        ASSERT_TRUE(dump);
    }
}

//
// IE Class GetConfig
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values("TEMPLATE"));

using IEClassGetConfigTestTEMPLATE = IEClassNetworkTest;

TEST_F(IEClassGetConfigTestTEMPLATE, nightly_GetConfigNoThrow) {
    Core ie;
    Parameter p;
    std::string deviceName = "TEMPLATE";

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto &&confKey : configValues) {
        if (CONFIG_KEY(DEVICE_ID) == confKey) {
            std::string defaultDeviceID = ie.GetConfig(deviceName, CONFIG_KEY(DEVICE_ID));
            std::cout << CONFIG_KEY(DEVICE_ID) << " : " << defaultDeviceID << std::endl;
        } else if (CONFIG_KEY(PERF_COUNT) == confKey) {
            bool defaultPerfCount = ie.GetConfig(deviceName, CONFIG_KEY(PERF_COUNT));
            std::cout << CONFIG_KEY(PERF_COUNT) << " : " << defaultPerfCount << std::endl;
        } else if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) == confKey) {
            bool defaultExclusive = ie.GetConfig(deviceName, CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS));
            std::cout << CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) << " : " << defaultExclusive << std::endl;
        }
    }
}

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("TEMPLATE", "MULTI:TEMPLATE", "HETERO:TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("TEMPLATE", "MULTI:TEMPLATE", "HETERO:TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("TEMPLATE", "MULTI:TEMPLATE", "HETERO:TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("TEMPLATE", "MULTI:TEMPLATE", "HETERO:TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported, IEClassExecutableNetworkGetMetricTest,
        ::testing::Values("TEMPLATE", "MULTI:TEMPLATE", "HETERO:TEMPLATE"));
//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values("TEMPLATE"));

// IE Class Query network

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values("TEMPLATE"));

// IE Class Load network

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values("TEMPLATE"));

//
// Hetero Executable Network GetMetric
//

#ifdef ENABLE_MKL_DNN

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("TEMPLATE"));

INSTANTIATE_TEST_CASE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values("TEMPLATE"));

#endif  // ENABLE_MKL_DNN
} // namespace