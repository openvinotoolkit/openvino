// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"

namespace ov {
namespace test {
namespace behavior {
std::string OVClassExecutableNetworkGetMetricTest_Priority::getTestCaseName(testing::TestParamInfo<PriorityParams> obj) {
    std::string target_device;
    ov::AnyMap userConfig;
    std::tie(target_device, userConfig) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    auto compileModelProperties = userConfig;
    std::ostringstream result;
    result << "device_name=" << target_device << "_";
    for (auto& iter : compileModelProperties) {
        result << iter.first << "_" << iter.second.as<std::string>() << "_";
    }
    return result.str();
}

void OVClassHeteroExecutableNetworkGetMetricTest::SetCpuAffinity(ov::Core& core, std::vector<std::string>& expectedTargets) {
#ifdef ENABLE_INTEL_CPU
    auto layermap = core.query_model(actualNetwork, heteroDeviceName);
    for (auto &iter : layermap) {
        if (iter.first.find("Concat") != std::string::npos)
            layermap[iter.first] = ov::test::utils::DEVICE_CPU;
    }
    for (auto& node : actualNetwork->get_ops()) {
        auto affinity = layermap[node->get_friendly_name()];
        node->get_rt_info()["affinity"] = affinity;
    }
    if (target_device.find(ov::test::utils::DEVICE_CPU) == std::string::npos)
        expectedTargets = {target_device, ov::test::utils::DEVICE_CPU};
#endif
}


//
// ImportExportNetwork
//

TEST_P(OVClassExecutableNetworkImportExportTestP, smoke_ImportNetworkNoThrowWithDeviceName) {
    ov::Core ie = createCoreWithTemplate();
    std::stringstream strm;
    ov::CompiledModel executableNetwork;
    OV_ASSERT_NO_THROW(executableNetwork = ie.compile_model(actualNetwork, target_device));
    OV_ASSERT_NO_THROW(executableNetwork.export_model(strm));
    OV_ASSERT_NO_THROW(executableNetwork = ie.import_model(strm, target_device));
    OV_ASSERT_NO_THROW(executableNetwork.create_infer_request());
}

//
// ExecutableNetwork GetMetric / GetConfig
//
TEST_P(OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

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

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

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

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::string model_name;
    OV_ASSERT_NO_THROW(model_name = compiled_model.get_property(ov::model_name));

    std::cout << "Compiled model name: " << std::endl << model_name << std::endl;
    ASSERT_EQ(simpleNetwork->get_friendly_name(), model_name);
    ASSERT_EXEC_METRIC_SUPPORTED(ov::model_name);
}

TEST_P(OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    unsigned int value = 0;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::optimal_number_of_infer_requests));

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED(ov::optimal_number_of_infer_requests);
}
TEST_P(OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    auto compiled_model = ie.compile_model(simpleNetwork, target_device, configuration);

    ov::hint::Priority value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::hint::model_priority));
    ASSERT_EQ(value, configuration[ov::hint::model_priority.name()].as<ov::hint::Priority>());
}

TEST_P(OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    auto compiled_model = ie.compile_model(simpleNetwork, target_device, configuration);

    std::string value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::device::priorities));
    ASSERT_EQ(value, configuration[ov::device::priorities.name()].as<std::string>());
}

TEST_P(OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported, GetMetricThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    ASSERT_THROW(compiled_model.get_property("unsupported_property"), ov::Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

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

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    ASSERT_THROW(compiled_model.get_property("unsupported_property"), ov::Exception);
}

TEST_P(OVClassExecutableNetworkSetConfigTest, SetConfigThrows) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    ASSERT_THROW(compiled_model.set_property({{"unsupported_config", "some_value"}}), ov::Exception);
}

TEST_P(OVClassExecutableNetworkSetConfigTest, canNotSetConfigToCompiledModelWithIncorrectConfig) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);
    std::map<std::string, std::string> incorrectConfig = {{"abc", "def"}};
    std::map<std::string, ov::Any> config;
    for (const auto& confItem : incorrectConfig) {
        config.emplace(confItem.first, confItem.second);
    }
    EXPECT_ANY_THROW(compiled_model.set_property(config));
}

TEST_P(OVClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    ov::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);
    OV_ASSERT_NO_THROW(compiled_model.set_property({{configKey, configValue}}));
    OV_ASSERT_NO_THROW(p = compiled_model.get_property(configKey));
    ASSERT_EQ(p, configValue);
}

TEST_P(OVClassExecutableNetworkGetMetricTestForSpecificConfig, canSetConfigToCompiledModel) {
    ov::Core ie = createCoreWithTemplate();
    std::shared_ptr<ov::Model> function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
    auto execNet = ie.compile_model(function, target_device);
    std::map<std::string, ov::Any> config;
    config.emplace(configKey, configValue);
    EXPECT_NO_THROW(execNet.set_property(config));
}

TEST_P(OVClassExecutableNetworkGetMetricTestForSpecificConfig, canSetConfigToCompiledModelGetConfigAndCheck) {
    ov::Core ie = createCoreWithTemplate();
    std::shared_ptr<ov::Model> function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
    auto execNet = ie.compile_model(simpleNetwork, target_device);
    std::map<std::string, ov::Any> config;
    config.emplace(configKey, configValue);
    execNet.set_property(config);
    ov::Any param;
    EXPECT_NO_THROW(param = execNet.get_property(configKey));
    EXPECT_FALSE(param.empty());
    EXPECT_EQ(param, configValue);
}

TEST_P(OVClassExecutableNetworkUnsupportedConfigTest, UnsupportedConfigThrows) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    ASSERT_THROW(compiled_model.set_property({{configKey, configValue}}), ov::Exception);
}

TEST_P(OVClassExecutableNetworkGetConfigTest, GetConfigNoEmptyNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::vector<ov::PropertyName> dev_property_names;
    OV_ASSERT_NO_THROW(dev_property_names = ie.get_property(target_device, ov::supported_properties));

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::vector<ov::PropertyName> model_property_names;
    OV_ASSERT_NO_THROW(model_property_names = compiled_model.get_property(ov::supported_properties));
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto heteroExeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);
    auto deviceExeNetwork = ie.compile_model(actualNetwork, target_device);

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

        if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) != deviceConf &&
            ov::supported_properties.name() != deviceConf) {
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
    auto deviceExeNetwork = ie.compile_model(actualNetwork, target_device);

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

    setHeteroNetworkAffinity(target_device);

    auto compiled_model = ie.compile_model(actualNetwork, heteroDeviceName);

    std::string targets;
    OV_ASSERT_NO_THROW(targets = compiled_model.get_property(ov::device::priorities));
    auto expectedTargets = target_device;

    std::cout << "Compiled model fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}

TEST_P(OVClassHeteroExecutableNetworkGetMetricTest_EXEC_DEVICES, GetMetricNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<std::string> expectedTargets = {target_device};

    SetCpuAffinity(ie, expectedTargets);

    auto compiled_model = ie.compile_model(actualNetwork, heteroDeviceName);

    std::vector<std::string> exeTargets;
    OV_ASSERT_NO_THROW(exeTargets = compiled_model.get_property(ov::execution_devices));

    ASSERT_EQ(expectedTargets, exeTargets);
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
