// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include <cstdint>

#include "openvino/runtime/properties.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

#include <locale.h>

namespace ov {
namespace test {
namespace behavior {

std::string OVClassCompiledModelEmptyPropertiesTests::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    return "target_device=" + obj.param;
}

void OVClassCompiledModelEmptyPropertiesTests::SetUp() {
    target_device = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
    model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
}

std::string OVClassCompiledModelPropertiesTests::getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
    std::string targetDevice;
    AnyMap properties;
    std::tie(targetDevice, properties) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
    std::ostringstream result;
    result << "targetDevice=" << targetDevice << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    return result.str();
}

void OVClassCompiledModelPropertiesTests::SetUp() {
    std::tie(target_device, properties) = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    APIBaseTest::SetUp();
    model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
}

void OVClassCompiledModelPropertiesTests::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

// check properties
std::string OVCompileModelGetExecutionDeviceTests::getTestCaseName(testing::TestParamInfo<OvPropertiesParams> obj) {
    std::string target_device;
    std::pair<ov::AnyMap, std::string> userConfig;
    std::tie(target_device, userConfig) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    auto compileModelProperties = userConfig.first;
    std::ostringstream result;
    result << "device_name=" << target_device << "_";
    if (!compileModelProperties.empty()) {
        result << "_compileModelProp=" << util::join(util::split(util::to_string(compileModelProperties), ' '), "_");
    }
    result << "_expectedDevice=" << userConfig.second;
    return result.str();
}

void OVCompileModelGetExecutionDeviceTests::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::pair<ov::AnyMap, std::string> userConfig;
    std::tie(target_device, userConfig) = GetParam();
    compileModelProperties = userConfig.first;
    expectedDeviceName = userConfig.second;
    model = ov::test::utils::make_conv_pool_relu();
}

TEST_P(OVClassCompiledModelPropertiesTests, CanUseCache) {
    std::string cache_dir = "./test_cache";
    core->set_property(ov::cache_dir(cache_dir));
    OV_ASSERT_NO_THROW(core->compile_model(model, target_device, properties));
    OV_ASSERT_NO_THROW(core->compile_model(model, target_device, properties));
    ov::test::utils::removeDir(cache_dir);
}

TEST_P(OVClassCompiledModelPropertiesTests, canCompileModelWithPropertiesAndCheckGetProperty) {
    auto compiled_model = core->compile_model(model, target_device, properties);
    auto supported_properties = compiled_model.get_property(ov::supported_properties);
    for (const auto& property_item : properties) {
        if (util::contains(supported_properties, property_item.first)) {
            Any property;
            OV_ASSERT_NO_THROW(property = compiled_model.get_property(property_item.first));
            ASSERT_FALSE(property.empty());
            std::cout << property_item.first << ":" << property.as<std::string>() << std::endl;
        }
    }
}

TEST_P(OVClassCompileModelWithCorrectPropertiesTest, IgnoreEnableMMap) {
    if (target_device.find("HETERO:") == 0 || target_device.find("MULTI:") == 0 || target_device.find("AUTO:") == 0 ||
        target_device.find("BATCH:") == 0)
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
    // Load available plugins
    core->get_available_devices();
    OV_ASSERT_NO_THROW(core->set_property(ov::enable_mmap(false)));
    OV_ASSERT_NO_THROW(core->set_property(target_device, ov::enable_mmap(false)));
}

TEST_P(OVClassCompileModelWithCorrectPropertiesTest, CompileModelWithCorrectPropertiesTest) {
    OV_ASSERT_NO_THROW(core->compile_model(model, target_device, properties));
}

TEST_P(OVClassCompiledModelPropertiesIncorrectTests, CanNotCompileModelWithIncorrectProperties) {
    ASSERT_THROW(core->compile_model(model, target_device, properties), ov::Exception);
}

TEST_P(OVCompiledModelIncorrectDevice, CanNotCompileModelWithIncorrectDeviceID) {
    ov::Core ie = ov::test::utils::create_core();
    ASSERT_THROW(ie.compile_model(actualNetwork, target_device + ".10"), ov::Exception);
}

TEST_P(OVCompiledModelIncorrectDevice, CanNotCompileModelWithEmpty) {
    ov::Core ie = ov::test::utils::create_core();
    ASSERT_THROW(ie.compile_model(actualNetwork, ""), ov::Exception);
}

TEST_P(OVCompiledModelPropertiesDefaultSupportedTests, CanCompileWithDefaultValueFromPlugin) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    AnyMap default_rw_properties;
    for (auto& supported_property : supported_properties) {
        if (supported_property.is_mutable()) {
            Any property;
            OV_ASSERT_NO_THROW(property = core->get_property(target_device, supported_property));
            default_rw_properties.emplace(supported_property, property);
            std::cout << supported_property << ":" << property.as<std::string>() << std::endl;
        }
    }
    OV_ASSERT_NO_THROW(core->compile_model(model, target_device, default_rw_properties));
}

TEST_P(OVClassCompiledModelPropertiesDefaultTests, CheckDefaultValues) {
    auto compiled_model = core->compile_model(model, target_device);
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = compiled_model.get_property(ov::supported_properties));
    std::cout << "SUPPORTED PROPERTIES: " << std::endl;
    for (auto&& supported_property : supported_properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = compiled_model.get_property(supported_property));
        std::cout << supported_property << ":" << property.as<std::string>() << std::endl;
    }
    for (auto&& default_property : properties) {
        auto supported = util::contains(supported_properties, default_property.first);
        ASSERT_TRUE(supported) << "default_property=" << default_property.first;
        Any property;
        OV_ASSERT_NO_THROW(property = compiled_model.get_property(default_property.first));
        ASSERT_EQ(default_property.second.as<std::string>(), property.as<std::string>())
            << "For property: " << default_property.first
            << " expected value is: " << default_property.second.as<std::string>();
    }
}

std::string OVClassCompiledModelGetPropertyTest_Priority::getTestCaseName(testing::TestParamInfo<PriorityParams> obj) {
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

// get property
TEST_P(OVClassCompiledModelGetConfigTest, GetConfigNoThrow) {
    ov::Core ie = ov::test::utils::create_core();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::vector<ov::PropertyName> property_names;
    OV_ASSERT_NO_THROW(property_names = compiled_model.get_property(ov::supported_properties));

    for (auto&& property : property_names) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = compiled_model.get_property(property));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassCompiledModelGetConfigTest, GetConfigFromCoreAndFromCompiledModel) {
    ov::Core ie = ov::test::utils::create_core();

    std::vector<ov::PropertyName> dev_property_names;
    OV_ASSERT_NO_THROW(dev_property_names = ie.get_property(target_device, ov::supported_properties));

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::vector<ov::PropertyName> model_property_names;
    OV_ASSERT_NO_THROW(model_property_names = compiled_model.get_property(ov::supported_properties));
}

// readonly
TEST_P(OVClassCompiledModelGetPropertyTest, GetMetricNoThrow_SUPPORTED_CONFIG_KEYS) {
    ov::Core ie = ov::test::utils::create_core();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = compiled_model.get_property(ov::supported_properties));

    std::cout << "Supported RW keys: " << std::endl;
    for (auto&& conf : supported_properties) {
        if (conf.is_mutable()) {
            std::cout << conf << std::endl;
            ASSERT_LT(0, conf.size());
        } else {
            std::cout << conf << std::endl;
            ASSERT_LT(0, conf.size());
        }
    }

    std::cout << "Supported RO keys: " << std::endl;
    for (auto&& conf : supported_properties) {
        if (!conf.is_mutable()) {
            std::cout << conf << std::endl;
            ASSERT_LT(0, conf.size());
        }
    }

    ASSERT_LE(0, supported_properties.size());
    ASSERT_EXEC_METRIC_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassCompiledModelGetPropertyTest, GetMetricNoThrow_NETWORK_NAME) {
    ov::Core ie = ov::test::utils::create_core();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::string model_name;
    OV_ASSERT_NO_THROW(model_name = compiled_model.get_property(ov::model_name));

    std::cout << "Compiled model name: " << std::endl << model_name << std::endl;
    ASSERT_EQ(simpleNetwork->get_friendly_name(), model_name);
    ASSERT_EXEC_METRIC_SUPPORTED(ov::model_name);
}

TEST_P(OVClassCompiledModelGetPropertyTest, GetMetricNoThrow_OPTIMAL_NUMBER_OF_INFER_REQUESTS) {
    ov::Core ie = ov::test::utils::create_core();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    unsigned int value = 0;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::optimal_number_of_infer_requests));

    std::cout << "Optimal number of Inference Requests: " << value << std::endl;
    ASSERT_GE(value, 1u);
    ASSERT_EXEC_METRIC_SUPPORTED(ov::optimal_number_of_infer_requests);
}

TEST_P(OVClassCompiledModelGetPropertyTest, CanCompileModelWithEmptyProperties) {
    ov::Core core = ov::test::utils::create_core();

    OV_ASSERT_NO_THROW(core.compile_model(simpleNetwork, target_device, ov::AnyMap{}));
}

TEST_P(OVClassCompiledModelGetIncorrectPropertyTest, GetConfigThrows) {
    ov::Core ie = ov::test::utils::create_core();
    auto compiled_model = ie.compile_model(simpleNetwork, target_device);
    ASSERT_THROW(compiled_model.get_property("unsupported_property"), ov::Exception);
}

// set property
TEST_P(OVClassCompiledModelSetCorrectConfigTest, canSetConfig) {
    ov::Core ie = ov::test::utils::create_core();
    ov::Any param;

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);
    OV_ASSERT_NO_THROW(compiled_model.set_property({{configKey, configValue}}));
    OV_ASSERT_NO_THROW(param = compiled_model.get_property(configKey));
    EXPECT_FALSE(param.empty());
    ASSERT_EQ(param, configValue);
}

TEST_P(OVClassCompiledModelSetIncorrectConfigTest, canNotSetConfigToCompiledModelWithIncorrectConfig) {
    ov::Core ie = ov::test::utils::create_core();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);
    std::map<std::string, std::string> incorrectConfig = {{"abc", "def"}};
    std::map<std::string, ov::Any> config;
    for (const auto& confItem : incorrectConfig) {
        config.emplace(confItem.first, confItem.second);
    }
    EXPECT_ANY_THROW(compiled_model.set_property(config));
}

// writeble
TEST_P(OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY, GetMetricNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    auto compiled_model = ie.compile_model(simpleNetwork, target_device, configuration);

    ov::hint::Priority value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::hint::model_priority));
    ASSERT_EQ(value, configuration[ov::hint::model_priority.name()].as<ov::hint::Priority>());
}

TEST_P(OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY, GetMetricNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    auto compiled_model = ie.compile_model(simpleNetwork, target_device, configuration);

    std::string value;
    OV_ASSERT_NO_THROW(value = compiled_model.get_property(ov::device::priorities));
    ASSERT_EQ(value, configuration[ov::device::priorities.name()].as<std::string>());
}

TEST_P(OVClassCompiledModelGetPropertyTest_EXEC_DEVICES, CanGetExecutionDeviceInfo) {
    ov::Core ie = ov::test::utils::create_core();
    std::vector<std::string> expectedTargets = {expectedDeviceName};
    auto compiled_model = ie.compile_model(model, target_device, compileModelProperties);

    std::vector<std::string> exeTargets;
    OV_ASSERT_NO_THROW(exeTargets = compiled_model.get_property(ov::execution_devices));

    ASSERT_EQ(expectedTargets, exeTargets);
}

TEST_P(OVCompileModelGetExecutionDeviceTests, CanGetExecutionDeviceInfo) {
    ov::CompiledModel exeNetWork;
    auto deviceList = core->get_available_devices();
    std::vector<std::string> expected_devices = util::split(expectedDeviceName, ',');
    std::vector<std::string> updatedExpectDevices;
    updatedExpectDevices.assign(expected_devices.begin(), expected_devices.end());
    for (auto& iter : compileModelProperties) {
        if ((iter.first == ov::hint::performance_mode &&
             iter.second.as<ov::hint::PerformanceMode>() == ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT) ||
            ov::test::behavior::sw_plugin_in_target_device(target_device)) {
            for (auto& deviceName : expected_devices) {
                for (auto&& device : deviceList) {
                    if (device.find(deviceName) != std::string::npos) {
                        auto updatedExpectDevices_iter =
                            std::find(updatedExpectDevices.begin(), updatedExpectDevices.end(), deviceName);
                        if (updatedExpectDevices_iter != updatedExpectDevices.end())
                            updatedExpectDevices.erase(updatedExpectDevices_iter);
                        updatedExpectDevices.push_back(std::move(device));
                    }
                }
            }
            break;
        }
    }
    std::sort(updatedExpectDevices.begin(), updatedExpectDevices.end());
    OV_ASSERT_NO_THROW(exeNetWork = core->compile_model(model, target_device, compileModelProperties));
    ov::Any property;
    OV_ASSERT_NO_THROW(property = exeNetWork.get_property(ov::execution_devices));
    std::vector<std::string> property_vector = property.as<std::vector<std::string>>();
    std::sort(property_vector.begin(), property_vector.end());
    if (expectedDeviceName.find("undefined") == std::string::npos)
        ASSERT_EQ(property_vector, updatedExpectDevices);
    else
        ASSERT_FALSE(property.empty());
}

TEST_P(OVClassCompiledModelGetConfigTest, CanCompileModelWithCustomLocale) {
    auto prev = std::locale().name();
    setlocale(LC_ALL, "en_GB.UTF-8");

    ov::Core core = ov::test::utils::create_core();

    OV_ASSERT_NO_THROW(core.compile_model(simpleNetwork, target_device););

    setlocale(LC_ALL, prev.c_str());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
