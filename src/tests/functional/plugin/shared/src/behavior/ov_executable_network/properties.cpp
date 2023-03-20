// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include <cstdint>

namespace ov {
namespace test {
namespace behavior {

std::string OVCompiledModelEmptyPropertiesTests::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    return "target_device=" + obj.param;
}

void OVCompiledModelEmptyPropertiesTests::SetUp() {
    target_device = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
    model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(target_device);
}

std::string OVCompiledModelPropertiesTests::getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
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

void OVCompiledModelPropertiesTests::SetUp() {
    std::tie(target_device, properties) = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    APIBaseTest::SetUp();
    model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(target_device);
}

void OVCompiledModelPropertiesTests::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

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

//
// Compile Model With Properties
//

TEST_P(OVCompiledModelEmptyPropertiesTests, CanCompileModelWithEmptyProperties) {
    OV_ASSERT_NO_THROW(core->compile_model(model, target_device, AnyMap{}));
}

TEST_P(OVCompiledModelPropertiesTests, CanCompileModelWithCorrectProperties) {
    OV_ASSERT_NO_THROW(core->compile_model(model, target_device, properties));
}

TEST_P(OVCompiledModelPropertiesTests, CanUseCache) {
    core->set_property(ov::cache_dir("./test_cache"));
    OV_ASSERT_NO_THROW(core->compile_model(model, target_device, properties));
    OV_ASSERT_NO_THROW(core->compile_model(model, target_device, properties));
    CommonTestUtils::removeDir("./test_cache");
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

TEST_P(OVCompiledModelPropertiesTests, canCompileModelWithPropertiesAndCheckGetProperty) {
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

TEST_P(OVCompiledModelPropertiesIncorrectTests, CanNotCompileModelWithIncorrectProperties) {
    ASSERT_THROW(core->compile_model(model, target_device, properties), ov::Exception);
}

TEST_P(OVCompiledModelPropertiesDefaultTests, CanCompileWithDefaultValueFromPlugin) {
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

TEST_P(OVCompiledModelPropertiesDefaultTests, CheckDefaultValues) {
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
        ASSERT_EQ(default_property.second, property) << "For property: " << default_property.first
            << " expected value is: " << default_property.second.as<std::string>();
    }
}

//
// Set Properties for Compiled Model
//

TEST_P(OVClassExecutableNetworkSupportedConfigTest, SupportedConfigWorks) {
    ov::Core ie = createCoreWithTemplate();
    ov::Any p;

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);
    OV_ASSERT_NO_THROW(compiled_model.set_property({{configKey, configValue}}));
    OV_ASSERT_NO_THROW(p = compiled_model.get_property(configKey));
    ASSERT_EQ(p, configValue);
}

TEST_P(OVClassCompiledModelUnsupportedConfigTest, UnsupportedConfigThrows) {
    ov::Core ie = createCoreWithTemplate();
    auto compiled_model = ie.compile_model(simpleNetwork, target_device);
    ASSERT_THROW(compiled_model.set_property({{configKey, configValue}}), ov::Exception);
}

//
// Get Properties from Compiled Model
//

TEST_P(OVClassCompiledModelProperties_SupportedProperties, checkSupportedRWProperties) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = compiled_model.get_property(ov::supported_properties));

    std::cout << "Supported RW keys: " << std::endl;
    for (auto&& conf : supported_properties)
        if (conf.is_mutable()) {
            std::cout << conf << std::endl;
            ASSERT_LT(0, conf.size());
        }
    ASSERT_LE(0, supported_properties.size());
    ASSERT_EXEC_METRIC_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassCompiledModelProperties_SupportedProperties, checkSupportedROProperties) {
    ov::Core ie = createCoreWithTemplate();

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = compiled_model.get_property(ov::supported_properties));

    std::cout << "Supported RO keys: " << std::endl;
    for (auto&& conf : supported_properties)
        if (!conf.is_mutable()) {
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

TEST_P(OVClassCompiledModelGetIncorrectProperties, failGetIncorrectProperty) {
    ov::Core ie = createCoreWithTemplate();
    auto compiled_model = ie.compile_model(simpleNetwork, target_device);
    ASSERT_THROW(compiled_model.get_property("unsupported_property"), ov::Exception);
}

TEST_P(OVCompiledModelGetSupportedPropertiesTest, canGetSupportedPropertiesAndDefaultValuesNotEmpty) {
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

TEST_P(OVCompiledModelGetSupportedPropertiesTest, canGetSupportedPropertiesBeforeAndAfterCopile) {
    ov::Core ie = createCoreWithTemplate();

    std::vector<ov::PropertyName> dev_property_names;
    OV_ASSERT_NO_THROW(dev_property_names = ie.get_property(target_device, ov::supported_properties));

    auto compiled_model = ie.compile_model(simpleNetwork, target_device);

    std::vector<ov::PropertyName> model_property_names;
    OV_ASSERT_NO_THROW(model_property_names = compiled_model.get_property(ov::supported_properties));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
