// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "base/behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

#include <iostream>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#include <codecvt>
#include <functional_test_utils/skip_tests_config.hpp>

#endif

namespace BehaviorTestsDefinitions {

#define ASSERT_EXEC_METRIC_SUPPORTED_IE(metricName)                  \
{                                                                    \
    auto metrics =                               \
        exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();         \
    auto it = std::find(metrics.begin(), metrics.end(), metricName); \
    ASSERT_NE(metrics.end(), it);                                    \
}

class IEClassExecutableNetworkGetMetricTestForSpecificConfig :
        public BehaviorTestsUtils::IEExecutableNetworkTestBase,
        public BehaviorTestsUtils::IEClassNetworkTest,
        public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>> {
protected:
    std::string configKey;
    std::string configValue;
public:
    void SetUp() override {
        target_device = std::get<0>(GetParam());
        std::tie(configKey, configValue) = std::get<1>(GetParam());
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        ov::test::behavior::APIBaseTest::SetUp();
        IEClassNetworkTest::SetUp();
    }
};

//
// Hetero Executable network case
//
class IEClassHeteroExecutableNetworkGetMetricTest :
        public BehaviorTestsUtils::IEExecutableNetworkTestBase,
        public BehaviorTestsUtils::IEClassNetworkTest,
        public ::testing::WithParamInterface<std::string> {
protected:
    std::string heteroDeviceName;
public:
    void SetUp() override {
        target_device = GetParam();
        heteroDeviceName = ov::test::utils::DEVICE_HETERO + std::string(":") + GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        ov::test::behavior::APIBaseTest::SetUp();
        IEClassNetworkTest::SetUp();
    }
};


//
// ImportExportNetwork
//

using IEClassGetMetricP = BehaviorTestsUtils::IEExecNetClassBaseTestP;
using IEClassImportExportTestP = IEClassGetMetricP;

TEST_P(IEClassImportExportTestP, smoke_ImportNetworkThrowsIfNoDeviceName) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    std::stringstream strm;
    InferenceEngine::ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualCnnNetwork, target_device));
    ASSERT_NO_THROW(executableNetwork.Export(strm));

    IE_SUPPRESS_DEPRECATED_START
    ASSERT_THROW(executableNetwork = ie.ImportNetwork(strm), InferenceEngine::Exception);
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(IEClassImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    std::stringstream strm;
    InferenceEngine::ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(actualCnnNetwork, target_device));
    ASSERT_NO_THROW(executableNetwork.Export(strm));
    ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(strm, target_device));
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
}

TEST_P(IEClassImportExportTestP, smoke_ExportUsingFileNameImportFromStreamNoThrowWithDeviceName) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::string fileName{"ExportedNetwork"};
    {
        ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device));
        ASSERT_NO_THROW(executableNetwork.Export(fileName));
    }
    {
        {
            std::ifstream strm(fileName, std::ifstream::binary | std::ifstream::in);
            ASSERT_NO_THROW(executableNetwork = ie.ImportNetwork(strm, target_device));
        }
        ASSERT_EQ(0, remove(fileName.c_str()));
    }
    ASSERT_NO_THROW(executableNetwork.CreateInferRequest());
    ov::test::utils::removeFile(fileName);
}

using IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassGetMetricP;
using IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = IEClassGetMetricP;
using IEClassExecutableNetworkGetMetricTest_NETWORK_NAME = IEClassGetMetricP;
using IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = IEClassGetMetricP;
using IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported = IEClassGetMetricP;
using IEClassExecutableNetworkGetConfigTest = IEClassGetMetricP;
using IEClassExecutableNetworkSetConfigTest = IEClassGetMetricP;

//
// ExecutableNetwork GetMetric / GetConfig
//
using IEClassExecutableNetworkSupportedConfigTest = IEClassExecutableNetworkGetMetricTestForSpecificConfig;
using IEClassExecutableNetworkUnsupportedConfigTest = IEClassExecutableNetworkGetMetricTestForSpecificConfig;

TEST_P(IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto configValues = p.as<std::vector<std::string>>();

    std::cout << "Supported config keys: " << std::endl;
    for (auto &&conf : configValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, configValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED_IE(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    auto metricValues = p.as<std::vector<std::string>>();

    std::cout << "Supported metric keys: " << std::endl;
    for (auto &&conf : metricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, metricValues.size());
    ASSERT_EXEC_METRIC_SUPPORTED_IE(METRIC_KEY(SUPPORTED_METRICS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    auto networkname = p.as<std::string>();

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
    ASSERT_EQ(simpleCnnNetwork.getName(), networkname);
    ASSERT_EXEC_METRIC_SUPPORTED_IE(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)));
    auto value = p.as<unsigned int>();

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED_IE(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
}

TEST_P(IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_THROW(p = exeNetwork.GetMetric("unsupported_metric"), InferenceEngine::Exception);
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto configValues = p.as<std::vector<std::string>>();

    for (auto &&confKey : configValues) {
        InferenceEngine::Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.GetConfig(target_device, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigThrows) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_THROW(p = exeNetwork.GetConfig("unsupported_config"), InferenceEngine::Exception);
}

TEST_P(IEClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_THROW(exeNetwork.SetConfig({{"unsupported_config", "some_value"}}), InferenceEngine::Exception);
}

TEST_P(IEClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(exeNetwork.SetConfig({{configKey, configValue}}));
    ASSERT_NO_THROW(p = exeNetwork.GetConfig(configKey));
    ASSERT_EQ(p, configValue);
}


TEST_P(IEClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_THROW(exeNetwork.SetConfig({{configKey, configValue}}), InferenceEngine::Exception);
}

TEST_P(IEClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto devConfigValues = p.as<std::vector<std::string>>();

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto execConfigValues = p.as<std::vector<std::string>>();

    /*
    for (auto && configKey : devConfigValues) {
        ASSERT_NE(execConfigValues.end(), std::find(execConfigValues.begin(), execConfigValues.end(), configKey));

        InferenceEngine::Parameter configValue;
        ASSERT_NO_THROW(Parameter configValue = exeNetwork.GetConfig(configKey));
    }
    */
}

using IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK = IEClassHeteroExecutableNetworkGetMetricTest;
using IEClassExecutableNetworkGetMetricTest = IEClassGetMetricP;

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter pHetero, pDevice;

    InferenceEngine::ExecutableNetwork heteroExeNetwork = ie.LoadNetwork(actualCnnNetwork, heteroDeviceName);
    InferenceEngine::ExecutableNetwork deviceExeNetwork = ie.LoadNetwork(actualCnnNetwork, target_device);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    auto heteroConfigValues = pHetero.as<std::vector<std::string>>(), deviceConfigValues = pDevice.as<std::vector<std::string>>();

    std::cout << "Supported config keys: " << std::endl;
    for (auto &&conf : heteroConfigValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, heteroConfigValues.size());

    // check that all device config values are unavailable in hetero case
    for (auto &&deviceConf : deviceConfigValues) {
        auto it = std::find(heteroConfigValues.begin(), heteroConfigValues.end(), deviceConf);
        if (it == heteroConfigValues.end()) {
            ASSERT_THROW(InferenceEngine::Parameter heteroConfigValue = heteroExeNetwork.GetConfig(deviceConf), InferenceEngine::Exception);
        }
    }
}

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter pHetero, pDevice;

    InferenceEngine::ExecutableNetwork heteroExeNetwork = ie.LoadNetwork(actualCnnNetwork, heteroDeviceName);
    InferenceEngine::ExecutableNetwork deviceExeNetwork = ie.LoadNetwork(actualCnnNetwork, target_device);

    ASSERT_NO_THROW(pHetero = heteroExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    ASSERT_NO_THROW(pDevice = deviceExeNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS)));
    auto heteroMetricValues = pHetero.as<std::vector<std::string>>(), deviceMetricValues = pDevice.as<std::vector<std::string>>();

    std::cout << "Supported metric keys: " << std::endl;
    for (auto &&conf : heteroMetricValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LT(0, heteroMetricValues.size());

    // check that all device metric values are unavailable in hetero case
    for (auto &&deviceMetricName : deviceMetricValues) {
        auto it = std::find(heteroMetricValues.begin(), heteroMetricValues.end(), deviceMetricName);
        if ((it == heteroMetricValues.end()) && (deviceMetricName != "SUPPORTED_PROPERTIES")) {
            ASSERT_THROW(InferenceEngine::Parameter heteroMetricValue = heteroExeNetwork.GetMetric(deviceMetricName), InferenceEngine::Exception);
        }
    }
}

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(actualCnnNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    auto networkname = p.as<std::string>();

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
}

TEST_P(IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;

    setHeteroNetworkAffinity(target_device);

    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(actualCnnNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.GetConfig("TARGET_FALLBACK"));
    auto targets = p.as<std::string>();
    auto expectedTargets = target_device;

    std::cout << "Exe network fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}
} // namespace BehaviorTestsDefinitions
