// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVClassSeveralDevicesTests : public OVPluginTestBase,
                                   public OVClassNetworkTest,
                                   public ::testing::WithParamInterface<std::vector<std::string>> {
public:
    std::vector<std::string> target_devices;

    void SetUp() override {
        target_device = CommonTestUtils::DEVICE_MULTI;
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
        target_devices = GetParam();
    }
};

using OVClassSeveralDevicesTestCompileModel = OVClassSeveralDevicesTests;
using OVClassSeveralDevicesTestQueryModel = OVClassSeveralDevicesTests;
using OVClassCompileModelWithCondidateDeviceListContainedMetaPluginTest = OVClassSetDevicePriorityConfigPropsTest;

TEST_P(OVClassCompileModelWithCondidateDeviceListContainedMetaPluginTest,
       CompileModelRepeatedlyWithMetaPluginTestThrow) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.compile_model(actualNetwork, target_device, configuration), ov::Exception);
}

TEST_P(OVClassSeveralDevicesTestCompileModel, CompileModelActualSeveralDevicesNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect DeviceID" << std::endl;

    std::string multitarget_device = CommonTestUtils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : target_devices) {
        multitarget_device += dev_name;
        if (&dev_name != &(target_devices.back())) {
            multitarget_device += ",";
        }
    }
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, multitarget_device));
}

TEST_P(OVClassModelTestP, CompileModelMultiWithoutSettingDevicePrioritiesThrows) {
    ov::Core ie = createCoreWithTemplate();
    try {
        ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_MULTI);
    } catch (ov::Exception& error) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            std::string("KEY_MULTI_DEVICE_PRIORITIES key is not set for"),
                            error.what());
    } catch (...) {
        FAIL() << "compile_model is failed for unexpected reason.";
    }
}

TEST_P(OVClassModelTestP, CompileModelActualHeteroDeviceUsingDevicePropertiesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork,
                                        CommonTestUtils::DEVICE_HETERO,
                                        ov::device::priorities(target_device),
                                        ov::device::properties(target_device, ov::enable_profiling(true))));
}

TEST_P(OVClassSeveralDevicesTestQueryModel, QueryNetworkActualSeveralDevicesNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    ASSERT_LT(deviceIDs.size(), target_devices.size());

    std::string multi_target_device = CommonTestUtils::DEVICE_MULTI + std::string(":");
    for (auto& dev_name : target_devices) {
        multi_target_device += dev_name;
        if (&dev_name != &(target_devices.back())) {
            multi_target_device += ",";
        }
    }
    OV_ASSERT_NO_THROW(ie.query_model(actualNetwork, multi_target_device));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov