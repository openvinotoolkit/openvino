// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/properties.hpp"
#include <cstdint>

namespace ov {
namespace test {
namespace behavior {

std::string OVEmptyPropertiesTests::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string target_device = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    return "target_device=" + target_device;
}

void OVEmptyPropertiesTests::SetUp() {
    target_device = this->GetParam();
    APIBaseTest::SetUp();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

std::string OVPropertiesTests::getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
    std::string target_device;
    AnyMap properties;
    std::tie(target_device, properties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    return result.str();
}

void OVPropertiesTests::SetUp() {
    std::tie(target_device, properties) = this->GetParam();
    APIBaseTest::SetUp();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

void OVPropertiesTests::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

std::string OVSetPropComplieModleGetPropTests::getTestCaseName(testing::TestParamInfo<CompileModelPropertiesParams> obj) {
    std::string target_device;
    AnyMap properties;
    AnyMap compileModelProperties;
    std::tie(target_device, properties, compileModelProperties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    if (!compileModelProperties.empty()) {
        result << "_compileModelProp=" << util::join(util::split(util::to_string(compileModelProperties), ' '), "_");
    }
    return result.str();
}

void OVSetPropComplieModleGetPropTests::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(target_device, properties, compileModelProperties) = this->GetParam();
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

TEST_P(OVEmptyPropertiesTests, SetEmptyProperties) {
    OV_ASSERT_NO_THROW(core->get_property(target_device, ov::supported_properties));
    OV_ASSERT_NO_THROW(core->set_property(target_device, AnyMap{}));
}

// Setting correct properties doesn't throw
TEST_P(OVPropertiesTests, SetCorrectProperties) {
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
}

TEST_P(OVPropertiesTests, canSetPropertyAndCheckGetProperty) {
    core->set_property(target_device, properties);
    for (const auto& property_item : properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(property.empty());
        std::cout << property_item.first << ":" << property.as<std::string>() << std::endl;
    }
}

TEST_P(OVPropertiesIncorrectTests, SetPropertiesWithIncorrectKey) {
    ASSERT_THROW(core->set_property(target_device, properties), ov::Exception);
}

TEST_P(OVPropertiesIncorrectTests, CanNotCompileModelWithIncorrectProperties) {
    ASSERT_THROW(core->compile_model(model, target_device, properties), ov::Exception);
}

TEST_P(OVPropertiesDefaultTests, CanSetDefaultValueBackToPlugin) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (auto& supported_property : supported_properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, supported_property));
        if (supported_property.is_mutable()) {
            OV_ASSERT_NO_THROW(core->set_property(target_device, {{ supported_property, property}}));
        }
    }
}

TEST_P(OVPropertiesDefaultTests, CheckDefaultValues) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (auto&& default_property : properties) {
        auto supported = util::contains(supported_properties, default_property.first);
        ASSERT_TRUE(supported) << "default_property=" << default_property.first;
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, default_property.first));
        ASSERT_EQ(default_property.second, property);
    }
}

TEST_P(OVSetPropComplieModleGetPropTests, SetPropertyComplieModelGetProperty) {
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));

    ov::CompiledModel exeNetWork;
    OV_ASSERT_NO_THROW(exeNetWork = core->compile_model(model, target_device, compileModelProperties));

    for (const auto& property_item : compileModelProperties) {
        Any exeNetProperty;
        OV_ASSERT_NO_THROW(exeNetProperty = exeNetWork.get_property(property_item.first));
        ASSERT_EQ(property_item.second.as<std::string>(), exeNetProperty.as<std::string>());
    }

    //the value of get property should be the same as set property
    for (const auto& property_item : properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, property_item.first));
        ASSERT_EQ(property_item.second.as<std::string>(), property.as<std::string>());
    }
}

TEST_P(OVSetPropComplieModleWihtIncorrectPropTests, SetPropertyComplieModelWithIncorrectProperty) {
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
    ASSERT_THROW(core->compile_model(model, target_device, compileModelProperties), ov::Exception);
}

TEST_P(OVSetPropComplieModleWihtIncorrectPropTests, CanNotCompileModelWithIncorrectProperties) {
    ASSERT_THROW(core->compile_model(model, target_device, properties), ov::Exception);
}

TEST_P(OVSetSupportPropComplieModleWithoutConfigTests, SetPropertyComplieModelWithCorrectProperty) {
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
    ASSERT_NO_THROW(core->compile_model(model, target_device, {}));
}

TEST_P(OVSetUnsupportPropComplieModleWithoutConfigTests, SetPropertyComplieModelWithIncorrectProperty) {
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
    ASSERT_THROW(core->compile_model(model, target_device, {}), ov::Exception);
}

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
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

TEST_P(OVCompileModelGetExecutionDeviceTests, CanGetExecutionDeviceInfo) {
    ov::CompiledModel exeNetWork;
    auto deviceList = core->get_available_devices();
    std::string updatedExpectDevices = expectedDeviceName;
    for (auto &iter : compileModelProperties) {
        if (iter.first == ov::hint::performance_mode && iter.second.as<ov::hint::PerformanceMode>() == ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT) {
            std::vector<std::string> expected_devices = util::split(expectedDeviceName, ',');
            std::vector<std::string> sameTypeDevices;
            for (auto& deviceName : expected_devices) {
                for (auto&& device : deviceList) {
                    if (device.find(deviceName) != std::string::npos) {
                        sameTypeDevices.push_back(std::move(device));
                    }
                }
            }
            updatedExpectDevices = util::join(sameTypeDevices, ",");
        }
    }
    OV_ASSERT_NO_THROW(exeNetWork = core->compile_model(model, target_device, compileModelProperties));
    ov::Any property;
    OV_ASSERT_NO_THROW(property = exeNetWork.get_property(ov::execution_devices));
    if (expectedDeviceName.find("undefined") == std::string::npos)
        ASSERT_EQ(util::join(property.as<std::vector<std::string>>(), ","), updatedExpectDevices);
    else
        ASSERT_FALSE(property.empty());
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
