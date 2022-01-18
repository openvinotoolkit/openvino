// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <base/ov_behavior_test_utils.hpp>

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>

#endif

namespace ov {
namespace test {
namespace behavior {

#define ASSERT_EXEC_METRIC_SUPPORTED(metricName)                                                \
    {                                                                                           \
        std::vector<std::string> metrics = exeNetwork.get_metric(METRIC_KEY(SUPPORTED_METRICS));\
        auto it = std::find(metrics.begin(), metrics.end(), metricName);                        \
        ASSERT_NE(metrics.end(), it);                                                           \
    }


using OVClassImportExportTestP = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_NETWORK_NAME = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported = OVClassBaseTestP;
using OVClassExecutableNetworkGetConfigTest = OVClassBaseTestP;
using OVClassExecutableNetworkSetConfigTest = OVClassBaseTestP;
using OVClassExecutableNetworkGetConfigTest = OVClassBaseTestP;

class OVClassExecutableNetworkGetMetricTestForSpecificConfig :
        public OVClassNetworkTest,
        public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>> {
protected:
    std::string deviceName;
    std::string configKey;
    std::string configValue;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVClassNetworkTest::SetUp();
        deviceName = std::get<0>(GetParam());
        std::tie(configKey, configValue) = std::get<1>(GetParam());
    }
};

using OVClassExecutableNetworkSupportedConfigTest = OVClassExecutableNetworkGetMetricTestForSpecificConfig;
using OVClassExecutableNetworkUnsupportedConfigTest = OVClassExecutableNetworkGetMetricTestForSpecificConfig;

//
// Hetero Executable network case
//
class OVClassHeteroExecutableNetworkGetMetricTest :
        public OVClassNetworkTest,
        public ::testing::WithParamInterface<std::string> {
protected:
    std::string deviceName;
    std::string heteroDeviceName;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVClassNetworkTest::SetUp();
        deviceName = GetParam();
        heteroDeviceName = CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName + std::string(",") +
                           CommonTestUtils::DEVICE_CPU;
    }
};
using OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK = OVClassHeteroExecutableNetworkGetMetricTest;

//
// ImportExportNetwork
//

TEST_P(OVClassImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    ov::runtime::Core ie = createCoreWithTemplate();
    std::stringstream strm;
    ov::runtime::CompiledModel executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.compile_model(actualNetwork, deviceName));
    ASSERT_NO_THROW(executableNetwork.export_model(strm));
    ASSERT_NO_THROW(executableNetwork = ie.import_model(strm, deviceName));
    ASSERT_NO_THROW(executableNetwork.create_infer_request());
}

//
// ExecutableNetwork GetMetric / GetConfig
//
TEST_P(OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    std::cout << "Supported config keys: " << std::endl;
    for (auto&& conf : configValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, configValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> metricValues = p;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto&& conf : metricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, metricValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
    ASSERT_EQ(simpleNetwork->get_friendly_name(), networkname);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)));
    unsigned int value = p;

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.get_metric("unsupported_metric"), ov::Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto&& confKey : configValues) {
        ov::Any defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.get_config(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.get_config("unsupported_config"), ov::Exception);
}

TEST_P(OVClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.set_config({{"unsupported_config", "some_value"}}), ov::Exception);
}

TEST_P(OVClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(exeNetwork.set_config({{configKey, configValue}}));
    ASSERT_NO_THROW(p = exeNetwork.get_config(configKey));
    ASSERT_EQ(p, configValue);
}

TEST_P(OVClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    ov::runtime::Core ie = createCoreWithTemplate();

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.set_config({{configKey, configValue}}), ov::Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> devConfigValues = p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> execConfigValues = p;

    /*
    for (auto && configKey : devConfigValues) {
        ASSERT_NE(execConfigValues.end(), std::find(execConfigValues.begin(), execConfigValues.end(), configKey));

        ov::Any configValue;
        ASSERT_NO_THROW(ov::Any configValue = exeNetwork.get_config(configKey));
    }
    */
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any pHetero, pDevice;

    auto heteroExeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);
    auto deviceExeNetwork = ie.compile_model(actualNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> heteroConfigValues = pHetero, deviceConfigValues = pDevice;

    std::cout << "Supported config keys: " << std::endl;
    for (auto&& conf : heteroConfigValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, heteroConfigValues.size());

    // check that all device config values are present in hetero case
    for (auto&& deviceConf : deviceConfigValues) {
        auto it = std::find(heteroConfigValues.begin(), heteroConfigValues.end(), deviceConf);
        ASSERT_TRUE(it != heteroConfigValues.end());

        ov::Any heteroConfigValue = heteroExeNetwork.get_config(deviceConf);
        ov::Any deviceConfigValue = deviceExeNetwork.get_config(deviceConf);

        // HETERO returns EXCLUSIVE_ASYNC_REQUESTS as a boolean value
        if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) != deviceConf) {
            ASSERT_EQ(deviceConfigValue, heteroConfigValue);
        }
    }
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any pHetero, pDevice;

    auto heteroExeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);
    auto deviceExeNetwork = ie.compile_model(actualNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.get_metric(METRIC_KEY(SUPPORTED_METRICS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.get_metric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> heteroMetricValues = pHetero, deviceMetricValues = pDevice;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto&& conf : heteroMetricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, heteroMetricValues.size());

    const std::vector<std::string> heteroSpecificMetrics = {METRIC_KEY(SUPPORTED_METRICS),
                                                            METRIC_KEY(SUPPORTED_CONFIG_KEYS)};

    // check that all device metric values are present in hetero case
    for (auto&& deviceMetricName : deviceMetricValues) {
        auto it = std::find(heteroMetricValues.begin(), heteroMetricValues.end(), deviceMetricName);
        ASSERT_TRUE(it != heteroMetricValues.end());

        ov::Any heteroMetricValue = heteroExeNetwork.get_metric(deviceMetricName);
        ov::Any deviceMetricValue = deviceExeNetwork.get_metric(deviceMetricName);

        if (std::find(heteroSpecificMetrics.begin(), heteroSpecificMetrics.end(), deviceMetricName) ==
            heteroSpecificMetrics.end()) {
            ASSERT_TRUE(heteroMetricValue == deviceMetricValue);
        }
    }
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto exeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    ov::runtime::Core ie = createCoreWithTemplate();
    ov::Any p;

    setHeteroNetworkAffinity(deviceName);

    auto exeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_config("TARGET_FALLBACK"));
    std::string targets = p;
    auto expectedTargets = deviceName + "," + CommonTestUtils::DEVICE_CPU;

    std::cout << "Exe network fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
