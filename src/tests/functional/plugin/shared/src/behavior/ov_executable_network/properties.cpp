// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"

#include <cstdint>

#include "openvino/runtime/properties.hpp"

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
    model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
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
    model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
}

void OVCompiledModelPropertiesTests::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

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
    ov::test::utils::removeDir("./test_cache");
}

TEST_P(OVCompiledModelPropertiesTests, IgnoreEnableMMap) {
    if (target_device.find("HETERO:") == 0 || target_device.find("MULTI:") == 0 || target_device.find("AUTO:") == 0 ||
        target_device.find("BATCH:") == 0)
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
    // Load available plugins
    core->get_available_devices();
    OV_ASSERT_NO_THROW(core->set_property(ov::enable_mmap(false)));
    OV_ASSERT_NO_THROW(core->set_property(target_device, ov::enable_mmap(false)));
}  // namespace behavior

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

TEST_P(OVClassCompileModelTest, LoadNetworkWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.compile_model(actualNetwork, target_device + ".10"), ov::Exception);
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
        ASSERT_EQ(default_property.second, property)
            << "For property: " << default_property.first
            << " expected value is: " << default_property.second.as<std::string>();
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
