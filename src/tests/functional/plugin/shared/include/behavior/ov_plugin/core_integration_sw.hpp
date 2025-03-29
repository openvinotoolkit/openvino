// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

using OVClassSeveralDevicesTestCompileModel = OVClassSeveralDevicesTests;
using OVClassSeveralDevicesTestQueryModel = OVClassSeveralDevicesTests;
using OVClassCompileModelWithCondidateDeviceListContainedMetaPluginTest = OVClassSetDevicePriorityConfigPropsTest;

TEST_P(OVClassCompileModelWithCondidateDeviceListContainedMetaPluginTest,
       CompileModelRepeatedlyWithMetaPluginTestThrow) {
    ov::Core ie = ov::test::utils::create_core();
    ASSERT_THROW(ie.compile_model(actualNetwork, target_device, configuration), ov::Exception);
}

TEST_P(OVClassSeveralDevicesTestCompileModel, CompileModelActualSeveralDevicesNoThrow) {
    ov::Core ie = ov::test::utils::create_core();

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect DeviceID" << std::endl;

    std::string multitarget_device = ov::test::utils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : target_devices) {
        multitarget_device += dev_name;
        if (&dev_name != &(target_devices.back())) {
            multitarget_device += ",";
        }
    }
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, multitarget_device));
}

TEST_P(OVClassModelOptionalTestP, CompileModelCreateDefaultExecGraphResult) {
    auto ie = ov::test::utils::create_core();
    auto net = ie.compile_model(actualNetwork, target_device);
    auto runtime_function = net.get_runtime_model();
    ASSERT_NE(nullptr, runtime_function);
    auto actual_parameters = runtime_function->get_parameters();
    auto actual_results = runtime_function->get_results();
    auto expected_parameters = actualNetwork->get_parameters();
    auto expected_results = actualNetwork->get_results();
    ASSERT_EQ(expected_parameters.size(), actual_parameters.size());
    for (std::size_t i = 0; i < expected_parameters.size(); ++i) {
        auto expected_element_type = expected_parameters[i]->get_output_element_type(0);
        auto actual_element_type = actual_parameters[i]->get_output_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_parameters[i]->get_output_shape(0);
        auto actual_shape = actual_parameters[i]->get_output_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
    ASSERT_EQ(expected_results.size(), actual_results.size());
    for (std::size_t i = 0; i < expected_results.size(); ++i) {
        auto expected_element_type = expected_results[i]->get_input_element_type(0);
        auto actual_element_type = actual_results[i]->get_input_element_type(0);
        ASSERT_EQ(expected_element_type, actual_element_type) << "For index: " << i;
        auto expected_shape = expected_results[i]->get_input_shape(0);
        auto actual_shape = actual_results[i]->get_input_shape(0);
        ASSERT_EQ(expected_shape, actual_shape) << "For index: " << i;
    }
}

TEST_P(OVClassSeveralDevicesTestQueryModel, QueryModelActualSeveralDevicesNoThrow) {
    ov::Core ie = ov::test::utils::create_core();

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    ASSERT_LE(deviceIDs.size(), target_devices.size());

    std::string multi_target_device = ov::test::utils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : target_devices) {
        multi_target_device += dev_name;
        if (&dev_name != &(target_devices.back())) {
            multi_target_device += ",";
        }
    }
    OV_ASSERT_NO_THROW(ie.query_model(actualNetwork, multi_target_device));
}

TEST(OVClassBasicPropsTest, smoke_SetConfigHeteroThrows) {
    ov::Core core;
    OV_ASSERT_NO_THROW(core.set_property(ov::test::utils::DEVICE_HETERO, ov::enable_profiling(true)));
}

TEST(OVClassBasicPropsTest, smoke_SetConfigDevicePropertiesThrows) {
    ov::Core core;
    ASSERT_THROW(core.set_property("", ov::device::properties(ov::test::utils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(core.set_property(ov::test::utils::DEVICE_CPU,
                                 ov::device::properties(ov::test::utils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(core.set_property(ov::test::utils::DEVICE_AUTO,
                                 ov::device::properties(ov::test::utils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(core.set_property(ov::test::utils::DEVICE_AUTO,
                                 ov::device::properties(ov::test::utils::DEVICE_CPU, ov::num_streams(4))),
                 ov::Exception);
}

TEST(OVClassBasicPropsTest, smoke_SetConfigAutoNoThrows) {
    ov::Core core;

    // priority config test
    // initialize, gcc 14.1 reports maybe-uninitialized at line 123 because of test macro at line 122
    ov::hint::Priority value{};
    OV_ASSERT_NO_THROW(core.set_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::LOW)));
    OV_ASSERT_NO_THROW(value = core.get_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::LOW);
    OV_ASSERT_NO_THROW(core.set_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::MEDIUM)));
    OV_ASSERT_NO_THROW(value = core.get_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::MEDIUM);
    OV_ASSERT_NO_THROW(core.set_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::HIGH)));
    OV_ASSERT_NO_THROW(value = core.get_property(ov::test::utils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::HIGH);
}

TEST(OVClassBasicPropsTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    ov::Core core;
    std::string target_device = ov::test::utils::DEVICE_HETERO;

    std::vector<ov::PropertyName> properties;
    OV_ASSERT_NO_THROW(properties = core.get_property(target_device, ov::supported_properties));

    std::cout << "Supported HETERO properties: " << std::endl;
    for (auto&& str : properties) {
        std::cout << str << " is_mutable: " << str.is_mutable() << std::endl;
    }

    auto it = std::find(properties.begin(), properties.end(), ov::supported_properties);
    ASSERT_NE(properties.end(), it);
}

TEST_P(OVClassModelOptionalTestP, getVersionsNonEmpty) {
    ov::Core core = ov::test::utils::create_core();
    ASSERT_EQ(2, core.get_versions(ov::test::utils::DEVICE_HETERO + std::string(":") + target_device).size());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
