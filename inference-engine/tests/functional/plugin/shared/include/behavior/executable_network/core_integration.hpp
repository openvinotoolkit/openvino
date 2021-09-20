// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <base/behavior_test_utils.hpp>
#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include <memory>
#include <fstream>
#include <ngraph/variant.hpp>
#include <functional_test_utils/plugin_cache.hpp>
#include <ngraph/op/util/op_types.hpp>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

#include <functional_test_utils/skip_tests_config.hpp>
#include <common_test_utils/common_utils.hpp>
#include <common_test_utils/test_assertions.hpp>

#ifdef ENABLE_UNICODE_PATH_SUPPORT

#include <iostream>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#include <codecvt>
#include <functional_test_utils/skip_tests_config.hpp>

#endif

using namespace testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace InferenceEngine::PluginConfigParams;

namespace BehaviorTestsDefinitions {
\

#define ASSERT_EXEC_METRIC_SUPPORTED_IE(metricName)                     \
{                                                                    \
    std::vector<std::string> metrics =                               \
        exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS));         \
    auto it = std::find(metrics.begin(), metrics.end(), metricName); \
    ASSERT_NE(metrics.end(), it);                                    \
}

class IEClassExecutableNetworkGetMetricTestForSpecificConfig : public BehaviorTestsUtils::IEClassNetworkTest,
                                                               public WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>> {
protected:
    std::string deviceName;
    std::string configKey;
    std::string configValue;
public:
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        deviceName = get<0>(GetParam());
        std::tie(configKey, configValue) = get<1>(GetParam());
    }
};

//
// Hetero Executable network case
//
class IEClassHeteroExecutableNetworkGetMetricTest : public BehaviorTestsUtils::IEClassNetworkTest,
                                                    public WithParamInterface<std::string> {
protected:
    std::string deviceName;
    std::string heteroDeviceName;
public:
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        deviceName = GetParam();
        heteroDeviceName = CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName + std::string(",") + CommonTestUtils::DEVICE_CPU;
    }
};


//
// ImportExportNetwork
//

using IEClassImportExportTestP = BehaviorTestsUtils::IEClassBaseTestP;

TEST_P(IEClassImportExportTestP, smoke_ImportNetworkThrowsIfNoDeviceName) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualCnnNetwork, deviceName));
    ASSERT_NO_THROW(executableNetwork.Export(strm));

    IE_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(executableNetwork = ie.ImportNetwork(strm), Exception);
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(IEClassImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    std::stringstream strm;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualCnnNetwork, deviceName));
    ASSERT_NO_THROW(executableNetwork.Export(strm));
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(strm, deviceName));
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

TEST_P(IEClassImportExportTestP, smoke_ExportUsingFileNameImportFromStreamNoThrowWithDeviceName) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    ExecutableNetwork executableNetwork;
    std::string fileName{"ExportedNetwork"};
    {
        ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName));
        ASSERT_NO_THROW(executableNetwork.Export(fileName));
    }
    {
        {
            std::ifstream strm(fileName);
            ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(strm, deviceName));
        }
        ASSERT_EQ(0, remove(fileName.c_str()));
    }
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

using IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_NETWORK_NAME = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassExecutableNetworkGetConfigTest = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassExecutableNetworkSetConfigTest = BehaviorTestsUtils::IEClassBaseTestP;
using IEClassExecutableNetworkGetConfigTest = BehaviorTestsUtils::IEClassBaseTestP;

//
// ExecutableNetwork GetMetric / GetConfig
//
using IEClassExecutableNetworkSupportedConfigTest = IEClassExecutableNetworkGetMetricTestForSpecificConfig;
using IEClassExecutableNetworkUnsupportedConfigTest = IEClassExecutableNetworkGetMetricTestForSpecificConfig;

TEST_P(IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    std::cout << "Supported config keys: " << std::endl;
    for (auto &&conf : configValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, configValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED_IE(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> metricValues = p;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto &&conf : metricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, metricValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED_IE(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
    ASSERT_EQ(simpleCnnNetwork.getName(), networkname);
    ASSERT_EXEC_METRIC_SUPPORTED_IE(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)));
    unsigned int value = p;

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED_IE(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.GetMetric("unsupported_metric"), Exception);
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto &&confKey : configValues) {
        Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.GetConfig(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigThrows) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.GetConfig("unsupported_config"), Exception);
}

TEST_P(IEClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_THROW(exeNetwork.SetConfig({{"unsupported_config", "some_value"}}), Exception);
}

TEST_P(IEClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(exeNetwork.SetConfig({{configKey, configValue}}));
    ASSERT_NO_THROW(p = exeNetwork.GetConfig(configKey));
    ASSERT_EQ(p, configValue);
}


TEST_P(IEClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_THROW(exeNetwork.SetConfig({{configKey, configValue}}), Exception);
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> devConfigValues = p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> execConfigValues = p;

    /*
    for (auto && configKey : devConfigValues) {
        ASSERT_NE(execConfigValues.end(), std::find(execConfigValues.begin(), execConfigValues.end(), configKey));

        Parameter configValue;
        ASSERT_NO_THROW(Parameter configValue = exeNetwork.GetConfig(configKey));
    }
    */
}

using IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassExecutableNetworkGetMetricTest = BehaviorTestsUtils::IEClassBaseTestP;

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter pHetero, pDevice;

    ExecutableNetwork heteroExeNetwork = ie.LoadNetwork(actualCnnNetwork, heteroDeviceName);
    ExecutableNetwork deviceExeNetwork = ie.LoadNetwork(actualCnnNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> heteroConfigValues = pHetero, deviceConfigValues = pDevice;

    std::cout << "Supported config keys: " << std::endl;
    for (auto &&conf : heteroConfigValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, heteroConfigValues.size());

    // check that all device config values are present in hetero case
    for (auto &&deviceConf : deviceConfigValues) {
        auto it = std::find(heteroConfigValues.begin(), heteroConfigValues.end(), deviceConf);
        ASSERT_TRUE(it != heteroConfigValues.end());

        Parameter heteroConfigValue = heteroExeNetwork.GetConfig(deviceConf);
        Parameter deviceConfigValue = deviceExeNetwork.GetConfig(deviceConf);

        // HETERO returns EXCLUSIVE_ASYNC_REQUESTS as a boolean value
        if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) != deviceConf) {
            ASSERT_EQ(deviceConfigValue, heteroConfigValue);
        }
    }
}

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter pHetero, pDevice;

    ExecutableNetwork heteroExeNetwork = ie.LoadNetwork(actualCnnNetwork, heteroDeviceName);
    ExecutableNetwork deviceExeNetwork = ie.LoadNetwork(actualCnnNetwork, deviceName);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    std::vector<std::string> heteroMetricValues = pHetero, deviceMetricValues = pDevice;

    std::cout << "Supported metric keys: " << std::endl;
    for (auto &&conf : heteroMetricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, heteroMetricValues.size());

    const std::vector<std::string> heteroSpecificMetrics = {
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
    };

    // check that all device metric values are present in hetero case
    for (auto &&deviceMetricName : deviceMetricValues) {
        auto it = std::find(heteroMetricValues.begin(), heteroMetricValues.end(), deviceMetricName);
        ASSERT_TRUE(it != heteroMetricValues.end());

        Parameter heteroMetricValue = heteroExeNetwork.GetMetric(deviceMetricName);
        Parameter deviceMetricValue = deviceExeNetwork.GetMetric(deviceMetricName);

        if (std::find(heteroSpecificMetrics.begin(), heteroSpecificMetrics.end(), deviceMetricName) ==
            heteroSpecificMetrics.end()) {
            ASSERT_TRUE(heteroMetricValue == deviceMetricValue);
        }
    }
}

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    ExecutableNetwork exeNetwork = ie.LoadNetwork(actualCnnNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
}

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    Parameter p;

    setHeteroNetworkAffinity(deviceName);

    ExecutableNetwork exeNetwork = ie.LoadNetwork(actualCnnNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetConfig("TARGET_FALLBACK"));
    std::string targets = p;
    auto expectedTargets = deviceName + "," + CommonTestUtils::DEVICE_CPU;

    std::cout << "Exe network fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}
} // namespace BehaviorTestsDefinitions
