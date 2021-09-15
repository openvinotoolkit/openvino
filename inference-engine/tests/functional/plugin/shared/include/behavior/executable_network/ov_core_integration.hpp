// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <base/behavior_test_utils.hpp>

#ifdef ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>

#endif

namespace BehaviorTestsDefinitions {

#define ASSERT_EXEC_METRIC_SUPPORTED(metricName)                                                \
    {                                                                                           \
        std::vector<std::string> metrics = exeNetwork.get_metric(METRIC_KEY(SUPPORTED_METRICS));\
        auto it = std::find(metrics.begin(), metrics.end(), metricName);                        \
        ASSERT_NE(metrics.end(), it);                                                           \
    }

using OVClassNetworkTestP = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassImportExportTestP = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_NETWORK_NAME = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassExecutableNetworkGetConfigTest = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassExecutableNetworkSetConfigTest = BehaviorTestsUtils::OVClassBaseTestP;
using OVClassExecutableNetworkGetConfigTest = BehaviorTestsUtils::OVClassBaseTestP;

class OVClassExecutableNetworkGetMetricTestForSpecificConfig :
        public BehaviorTestsUtils::OVClassNetworkTest,
        public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>> {
protected:
    std::string deviceName;
    std::string configKey;
    std::string configValue;

public:
    void SetUp() override {
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
        public BehaviorTestsUtils::OVClassNetworkTest,
        public ::testing::WithParamInterface<std::string> {
protected:
    std::string deviceName;
    std::string heteroDeviceName;

public:
    void SetUp() override {
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

bool supportsAvaliableDevices(ov::runtime::Core& ie, const std::string& deviceName) {
    auto supportedMetricKeys = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
    return supportedMetricKeys.end() !=
           std::find(std::begin(supportedMetricKeys), std::end(supportedMetricKeys), METRIC_KEY(AVAILABLE_DEVICES));
}
//
// LoadNetwork
//

TEST_P(OVClassNetworkTestP, LoadNetworkActualNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, deviceName));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + deviceName));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, {{"TARGET_FALLBACK", deviceName}}));
}

TEST_P(OVClassNetworkTestP, LoadNetworkCreateDefaultExecGraphResult) {
    auto ie = BehaviorTestsUtils::createCoreWithTemplate();
    auto net = ie.compile_model(actualNetwork, deviceName);
    auto runtime_function = net.get_runtime_function();
    ASSERT_NE(nullptr, runtime_function);
    auto actual_parameters = runtime_function->get_parameters();
    auto actual_results = runtime_function->get_results();
    auto expected_parameters = actualNetwork->get_parameters();
    auto expected_results = actualNetwork->get_results();
    ASSERT_EQ(expected_parameters.size(), actual_parameters.size());
    for (std::size_t i = 0; i < expected_parameters.size(); ++i) {
        auto expected_element_type = expected_parameters[i]->get_output_element_type(0);
        auto actual_element_type = actual_parameters[i]->get_output_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_parameters[i]->get_output_shape(0);
        auto actual_shape = actual_parameters[i]->get_output_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
    ASSERT_EQ(expected_results.size(), actual_results.size());
    for (std::size_t i = 0; i < expected_results.size(); ++i) {
        auto expected_element_type = expected_results[i]->get_input_element_type(0);
        auto actual_element_type = actual_results[i]->get_input_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_results[i]->get_input_shape(0);
        auto actual_shape = actual_results[i]->get_input_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
}

//
// ImportExportNetwork
//

TEST_P(OVClassImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    std::stringstream strm;
    ov::runtime::ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.compile_model(actualNetwork, deviceName));
    ASSERT_NO_THROW(executableNetwork.export_model(strm));
    ASSERT_NO_THROW(executableNetwork = ie.import_model(strm, deviceName));
    ASSERT_NO_THROW(executableNetwork.create_infer_request());
}

//
// ExecutableNetwork GetMetric / GetConfig
//
TEST_P(OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

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
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

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
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
    ASSERT_EQ(simpleNetwork->get_friendly_name(), networkname);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)));
    unsigned int value = p;

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED(EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
}

TEST_P(OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.get_metric("unsupported_metric"), InferenceEngine::Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto&& confKey : configValues) {
        InferenceEngine::Parameter defaultValue;
        ASSERT_NO_THROW(defaultValue = ie.get_config(deviceName, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigThrows) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(p = exeNetwork.get_config("unsupported_config"), InferenceEngine::Exception);
}

TEST_P(OVClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.set_config({{"unsupported_config", "some_value"}}), InferenceEngine::Exception);
}

TEST_P(OVClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(exeNetwork.set_config({{configKey, configValue}}));
    ASSERT_NO_THROW(p = exeNetwork.get_config(configKey));
    ASSERT_EQ(p, configValue);
}

TEST_P(OVClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(exeNetwork.set_config({{configKey, configValue}}), InferenceEngine::Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.get_metric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> devConfigValues = p;

    auto exeNetwork = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> execConfigValues = p;

    /*
    for (auto && configKey : devConfigValues) {
        ASSERT_NE(execConfigValues.end(), std::find(execConfigValues.begin(), execConfigValues.end(), configKey));

        InferenceEngine::Parameter configValue;
        ASSERT_NO_THROW(Parameter configValue = exeNetwork.get_config(configKey));
    }
    */
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter pHetero, pDevice;

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

        InferenceEngine::Parameter heteroConfigValue = heteroExeNetwork.get_config(deviceConf);
        InferenceEngine::Parameter deviceConfigValue = deviceExeNetwork.get_config(deviceConf);

        // HETERO returns EXCLUSIVE_ASYNC_REQUESTS as a boolean value
        if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) != deviceConf) {
            ASSERT_EQ(deviceConfigValue, heteroConfigValue);
        }
    }
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter pHetero, pDevice;

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

        InferenceEngine::Parameter heteroMetricValue = heteroExeNetwork.get_metric(deviceMetricName);
        InferenceEngine::Parameter deviceMetricValue = deviceExeNetwork.get_metric(deviceMetricName);

        if (std::find(heteroSpecificMetrics.begin(), heteroSpecificMetrics.end(), deviceMetricName) ==
            heteroSpecificMetrics.end()) {
            ASSERT_TRUE(heteroMetricValue == deviceMetricValue);
        }
    }
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    auto exeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_metric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)));
    std::string networkname = p;

    std::cout << "Exe network name: " << std::endl << networkname << std::endl;
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    ov::runtime::Core ie = BehaviorTestsUtils::createCoreWithTemplate();
    InferenceEngine::Parameter p;

    setHeteroNetworkAffinity(deviceName);

    auto exeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);

    ASSERT_NO_THROW(p = exeNetwork.get_config("TARGET_FALLBACK"));
    std::string targets = p;
    auto expectedTargets = deviceName + "," + CommonTestUtils::DEVICE_CPU;

    std::cout << "Exe network fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}

}  // namespace BehaviorTestsDefinitions
