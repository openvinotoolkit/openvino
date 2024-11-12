//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/ov_plugin/internal_properties_tests.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace ov::test::behavior {

std::string OVPropertiesTestsNPU::getTestCaseName(testing::TestParamInfo<PropertiesParamsNPU> obj) {
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

void OVPropertiesTestsNPU::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(target_device, properties) = this->GetParam();
    APIBaseTest::SetUp();
    model = ov::test::utils::make_split_concat();
}

void OVPropertiesTestsNPU::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

std::string OVPropertiesTestsWithCompileModelPropsNPU::getTestCaseName(
    testing::TestParamInfo<PropertiesParamsNPU> obj) {
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

void OVPropertiesTestsWithCompileModelPropsNPU::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::string temp_device;
    std::tie(temp_device, properties) = this->GetParam();
    std::string::size_type pos = temp_device.find(":", 0);
    std::string hw_device;

    if (pos == std::string::npos) {
        target_device = temp_device;
        hw_device = temp_device;
    } else {
        target_device = temp_device.substr(0, pos);
        hw_device = temp_device.substr(++pos, std::string::npos);
    }

    if (target_device == std::string(ov::test::utils::DEVICE_MULTI) ||
        target_device == std::string(ov::test::utils::DEVICE_AUTO) ||
        target_device == std::string(ov::test::utils::DEVICE_HETERO) ||
        target_device == std::string(ov::test::utils::DEVICE_BATCH)) {
        compileModelProperties = {ov::device::priorities(hw_device)};
    }

    model = ov::test::utils::make_split_concat();

    APIBaseTest::SetUp();
}

void OVPropertiesTestsWithCompileModelPropsNPU::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

TEST_P(OVPropertiesTestsNPU, SetCorrectProperties) {
    core->get_versions(target_device);
    core->set_property(target_device, properties);
}

TEST_P(OVPropertiesTestsNPU, canSetPropertyAndCheckGetProperty) {
    core->set_property(target_device, properties);

    for (const auto& property_item : properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(property.empty()) << property_item.first << ":" << property.as<std::string>() << '\n';
    }
}

TEST_P(OVPropertiesIncorrectTestsNPU, SetPropertiesWithIncorrectKey) {
    core->get_versions(target_device);
    std::vector<ov::PropertyName> supported_properties;
    supported_properties = core->get_property(target_device, ov::supported_properties);

    for (const auto& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_FALSE(supported);
    }
}

TEST_P(OVCheckSetSupportedRWMetricsPropsTestsNPU, ChangeCorrectProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));

    for (const auto& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_FALSE(supported);

        ov::Any default_property;
        OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(default_property.empty());

        if (ov::PropertyName(property_item.first).is_mutable() && !property_item.second.empty()) {
            OV_ASSERT_NO_THROW(core->set_property(target_device, {property_item}));
            core->compile_model(model, target_device, compileModelProperties);
            ov::Any actual_property_value;
            OV_ASSERT_NO_THROW(actual_property_value = core->get_property(target_device, property_item.first));
            ASSERT_FALSE(actual_property_value.empty());

            std::string expect_value = property_item.second.as<std::string>();
            std::string actual_value = actual_property_value.as<std::string>();
            EXPECT_EQ(actual_value, expect_value) << "Property is changed in wrong way";
        }
    }
}

const std::vector<ov::AnyMap> CorrectPluginMutableProperties = {
    {{ov::internal::exclusive_async_requests.name(), true}},
    {{ov::intel_npu::dpu_groups.name(), 1}},
    {{ov::intel_npu::dma_engines.name(), 1}},
    {{ov::intel_npu::compilation_mode.name(), "DefaultHW"}},
    {{ov::intel_npu::platform.name(),
      removeDeviceNameOnlyID(
          ov::test::utils::getTestsDeviceNameFromEnvironmentOr(std::string(ov::intel_npu::Platform::AUTO_DETECT)))}},
    {{ov::intel_npu::stepping.name(), 0}},
    {{ov::intel_npu::profiling_type.name(), ov::intel_npu::ProfilingType::INFER}}};

const std::vector<ov::AnyMap> IncorrectMutablePropertiesWrongValueTypes = {
    {{ov::intel_npu::compilation_mode.name(), -3.6}},
    {{ov::intel_npu::stepping.name(), "V1"}},
    {{ov::intel_npu::profiling_type.name(), 10}},
    {{ov::intel_npu::dma_engines.name(), false}},
};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVPropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(CorrectPluginMutableProperties)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesTestsNPU>));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVPropertiesIncorrectTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(IncorrectMutablePropertiesWrongValueTypes)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesIncorrectTestsNPU>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCheckSetSupportedRWMetricsPropsTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(CorrectPluginMutableProperties)),
                         (ov::test::utils::appendPlatformTypeTestName<OVCheckSetSupportedRWMetricsPropsTestsNPU>));
}  // namespace ov::test::behavior
