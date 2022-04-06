// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include <cstdint>

namespace ov {
namespace test {
namespace behavior {

std::string OVCompiledModelEmptyPropertiesTests::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    return "device_name=" + obj.param;
}

void OVCompiledModelEmptyPropertiesTests::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    device_name = this->GetParam();
    model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(device_name);
}

std::string OVCompiledModelPropertiesTests::getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
    std::string device_name;
    AnyMap properties;
    std::tie(device_name, properties) = obj.param;
    std::ostringstream result;
    result << "device_name=" << device_name << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    return result.str();
}

void OVCompiledModelPropertiesTests::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(device_name, properties) = this->GetParam();
    model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(device_name);
}

void OVCompiledModelPropertiesTests::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
}

TEST_P(OVCompiledModelEmptyPropertiesTests, CanCompileModelWithEmptyProperties) {
    OV_ASSERT_NO_THROW(core->compile_model(model, device_name, AnyMap{}));
}

TEST_P(OVCompiledModelPropertiesTests, CanCompileModelWithCorrectProperties) {
    OV_ASSERT_NO_THROW(core->compile_model(model, device_name, properties));
}

TEST_P(OVCompiledModelPropertiesTests, CanUseCache) {
    core->set_property(ov::cache_dir("./test_cache"));
    OV_ASSERT_NO_THROW(core->compile_model(model, device_name, properties));
    OV_ASSERT_NO_THROW(core->compile_model(model, device_name, properties));
    CommonTestUtils::removeDir("./test_cache");
}

TEST_P(OVCompiledModelPropertiesTests, canCompileModelWithPropertiesAndCheckGetProperty) {
    auto compiled_model = core->compile_model(model, device_name, properties);
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
    ASSERT_THROW(core->compile_model(model, device_name, properties), ov::Exception);
}

TEST_P(OVCompiledModelPropertiesDefaultTests, CanCompileWithDefaultValueFromPlugin) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(device_name, ov::supported_properties));
    AnyMap default_rw_properties;
    for (auto& supported_property : supported_properties) {
        if (supported_property.is_mutable()) {
            Any property;
            OV_ASSERT_NO_THROW(property = core->get_property(device_name, supported_property));
            default_rw_properties.emplace(supported_property, property);
            std::cout << supported_property << ":" << property.as<std::string>() << std::endl;
        }
    }
    OV_ASSERT_NO_THROW(core->compile_model(model, device_name, default_rw_properties));
}

TEST_P(OVCompiledModelPropertiesDefaultTests, CheckDefaultValues) {
    auto compiled_model = core->compile_model(model, device_name);
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
        ASSERT_EQ(default_property.second, property);
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov