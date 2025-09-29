//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal_properties_tests.hpp"

#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "overload/ov_plugin/properties_tests.hpp"

namespace ov::test::behavior {

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

const std::vector<ov::AnyMap> compat_CorrectPluginMutableProperties = {
    {{ov::internal::exclusive_async_requests.name(), true}},
    {{ov::intel_npu::dma_engines.name(), 1}},
    {{ov::intel_npu::compilation_mode.name(), "DefaultHW"}},
    {{ov::intel_npu::platform.name(),
      removeDeviceNameOnlyID(
          ov::test::utils::getTestsDeviceNameFromEnvironmentOr(std::string(ov::intel_npu::Platform::AUTO_DETECT)))}},
    {{ov::intel_npu::profiling_type.name(), ov::intel_npu::ProfilingType::INFER}}};

const std::vector<ov::AnyMap> compat_IncorrectMutablePropertiesWrongValueTypes = {
    {{ov::intel_npu::compilation_mode.name(), -3.6}},
    {{ov::intel_npu::profiling_type.name(), 10}},
    {{ov::intel_npu::dma_engines.name(), false}},
};

const std::vector<ov::AnyMap> CorrectPluginMutableProperties = {
    {{ov::intel_npu::stepping.name(), 0}},
};

const std::vector<ov::AnyMap> IncorrectMutablePropertiesWrongValueTypes = {
    {{ov::intel_npu::stepping.name(), "V1"}},
};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVPropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compat_CorrectPluginMutableProperties)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesTestsNPU>));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVPropertiesIncorrectTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compat_IncorrectMutablePropertiesWrongValueTypes)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesIncorrectTestsNPU>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCheckSetSupportedRWMetricsPropsTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::ValuesIn([] {
                                                std::vector<ov::AnyMap> combined =
                                                    compat_CorrectPluginMutableProperties;
                                                combined.insert(combined.end(),
                                                                CorrectPluginMutableProperties.begin(),
                                                                CorrectPluginMutableProperties.end());
                                                return combined;
                                            }())),
                         (ov::test::utils::appendPlatformTypeTestName<OVCheckSetSupportedRWMetricsPropsTestsNPU>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesIncorrectTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(IncorrectMutablePropertiesWrongValueTypes)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesIncorrectTestsNPU>));

}  // namespace ov::test::behavior
