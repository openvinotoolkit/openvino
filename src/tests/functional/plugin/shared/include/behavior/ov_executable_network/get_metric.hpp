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

#define ASSERT_EXEC_METRIC_SUPPORTED(property)                                                \
    {                                                                                           \
        auto properties = compiled_model.get_property(ov::supported_properties);\
        auto it = std::find(properties.begin(), properties.end(), property);                        \
        ASSERT_NE(properties.end(), it);                                                           \
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
    ov::Any configValue;

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
    ov::Core ie = createCoreWithTemplate();
    std::stringstream strm;
    ov::CompiledModel executableNetwork;
    OV_ASSERT_NO_THROW(executableNetwork = ie.compile_model(actualNetwork, deviceName));
    OV_ASSERT_NO_THROW(executableNetwork.export_model(strm));
    OV_ASSERT_NO_THROW(executableNetwork = ie.import_model(strm, deviceName));
    OV_ASSERT_NO_THROW(executableNetwork.create_infer_request());
}

//
// ExecutableNetwork GetMetric / GetConfig
//
TEST_P(OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = compiled_model.get_property(ov::supported_properties));

    std::cout << "Supported RW keys: " << std::endl;
    for (auto&& conf : supported_properties) if (conf.is_mutable()) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, supported_properties.size());
    ASSERT_EXEC_METRIC_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = compiled_model.get_property(ov::supported_properties));

    std::cout << "Supported RO keys: " << std::endl;
    for (auto&& conf : supported_properties) if (!conf.is_mutable()) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, supported_properties.size());
    ASSERT_EXEC_METRIC_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    std::string model_name;
    OV_ASSERT_NO_THROW(model_name = compiled_model.get_property(ov::model_name));

    std::cout << "Compiled model name: " << std::endl << model_name << std::endl;
    ASSERT_EQ(simpleNetwork->get_friendly_name(), model_name);
    ASSERT_EXEC_METRIC_SUPPORTED(ov::model_name);
}

TEST_P(OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    unsigned int value = 0;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::optimal_number_of_infer_requests));

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED(ov::optimal_number_of_infer_requests);
}
TEST_P(OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    auto compiled_model = ie.compile_model(simpleNetwork, deviceName, configuration);

    ov::hint::Priority value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::hint::model_priority));
    ASSERT_EQ(value, configuration[ov::hint::model_priority.name()].as<ov::hint::Priority>());
}

TEST_P(OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    auto compiled_model = ie.compile_model(simpleNetwork, deviceName, configuration);

    std::string value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::device::priorities));
    ASSERT_EQ(value, configuration[ov::device::priorities.name()].as<std::string>());
}

TEST_P(OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(compiled_model.get_property("unsupported_property"), ov::Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    std::vector<ov::PropertyName> property_names;
    OV_ASSERT_NO_THROW(property_names = compiled_model.get_property(ov::supported_properties));

    for (auto&& property : property_names) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = compiled_model.get_property(property));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigThrows) {
    ov::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(compiled_model.get_property("unsupported_property"), ov::Exception);
}

TEST_P(OVClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(compiled_model.set_property({{"unsupported_config", "some_value"}}), ov::Exception);
}

TEST_P(OVClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    ov::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);
    OV_ASSERT_NO_THROW(compiled_model.set_property({{configKey, configValue}}));
    OV_ASSERT_NO_THROW(p = compiled_model.get_property(configKey));
    ASSERT_EQ(p, configValue);
}

TEST_P(OVClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_THROW(compiled_model.set_property({{configKey, configValue}}), ov::Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::vector<ov::PropertyName> dev_property_names;
    OV_ASSERT_NO_THROW(dev_property_names = ie.get_property(deviceName, ov::supported_properties));

    auto compiled_model = ie.compile_model(simpleNetwork, deviceName);

    std::vector<ov::PropertyName> model_property_names;
    OV_ASSERT_NO_THROW(model_property_names = compiled_model.get_property(ov::supported_properties));
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto heteroExeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);
    auto deviceExeNetwork = ie.compile_model(actualNetwork, deviceName);

    std::vector<ov::PropertyName> heteroConfigValues, deviceConfigValues;
    OV_ASSERT_NO_THROW(heteroConfigValues = heteroExeNetwork.get_property(ov::supported_properties));
    OV_ASSERT_NO_THROW(deviceConfigValues = deviceExeNetwork.get_property(ov::supported_properties));

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

        ov::Any heteroConfigValue = heteroExeNetwork.get_property(deviceConf);
        ov::Any deviceConfigValue = deviceExeNetwork.get_property(deviceConf);

        // HETERO returns EXCLUSIVE_ASYNC_REQUESTS as a boolean value
        if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) != deviceConf) {
            std::stringstream strm;
            deviceConfigValue.print(strm);
            strm << " ";
            heteroConfigValue.print(strm);
            ASSERT_EQ(deviceConfigValue, heteroConfigValue) << deviceConf << " " << strm.str();
        }
    }
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto heteroExeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);
    auto deviceExeNetwork = ie.compile_model(actualNetwork, deviceName);

    std::vector<ov::PropertyName> heteroConfigValues, deviceConfigValues;
    OV_ASSERT_NO_THROW(heteroConfigValues = heteroExeNetwork.get_property(ov::supported_properties));
    OV_ASSERT_NO_THROW(deviceConfigValues = deviceExeNetwork.get_property(ov::supported_properties));

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

        ov::Any heteroConfigValue = heteroExeNetwork.get_property(deviceConf);
        ov::Any deviceConfigValue = deviceExeNetwork.get_property(deviceConf);

        // HETERO returns EXCLUSIVE_ASYNC_REQUESTS as a boolean value
        if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) != deviceConf) {
            std::stringstream strm;
            deviceConfigValue.print(strm);
            strm << " ";
            heteroConfigValue.print(strm);
            ASSERT_EQ(deviceConfigValue, heteroConfigValue) << deviceConf << " " << strm.str();
        }
    }
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(actualNetwork, heteroDeviceName);

    std::string model_name;
    OV_ASSERT_NO_THROW(model_name = compiled_model.get_property(ov::model_name));

    std::cout << "Compiled model name: " << std::endl << model_name << std::endl;
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    setHeteroNetworkAffinity(deviceName);

    auto compiled_model = ie.compile_model(actualNetwork, heteroDeviceName);

    std::string targets;
    OV_ASSERT_NO_THROW(targets = compiled_model.get_property(ov::device::priorities));
    auto expectedTargets = deviceName + "," + CommonTestUtils::DEVICE_CPU;

    std::cout << "Compiled model fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
