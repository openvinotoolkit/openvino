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
    return "target_device=" + obj.param;
}

void OVEmptyPropertiesTests::SetUp() {
    target_device = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

std::string OVPropertiesTests::getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
    std::string target_device;
    AnyMap properties;
    std::tie(target_device, properties) = obj.param;
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    return result.str();
}

void OVPropertiesTests::SetUp() {
    std::tie(target_device, properties) = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

void OVPropertiesTests::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
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

}  // namespace behavior
}  // namespace test
}  // namespace ov
