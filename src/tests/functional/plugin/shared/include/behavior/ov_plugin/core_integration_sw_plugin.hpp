// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "behavior/ov_plugin/core_integration.hpp"
#include <openvino/runtime/properties.hpp>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/util/file_util.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>
#endif

namespace ov {
namespace test {
namespace behavior {

using OVClassNetworkTestP = OVClassBaseTestP;
using OVClassQueryNetworkTest = OVClassBaseTestP;
using OVClassLoadNetworkTest = OVClassBaseTestP;

using OVClassGetConfigTest = OVClassBaseTestP;
using OVClassGetConfigTest_ThrowUnsupported = OVClassBaseTestP;
using OVClassLoadNetworkAfterCoreRecreateTest = OVClassBaseTestP;


class OVClassSeveralDevicesTest : public OVPluginTestBase,
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

using OVClassSeveralDevicesTestQueryNetwork = OVClassSeveralDevicesTest;
using OVClassSeveralDevicesTestLoadNetwork = OVClassSeveralDevicesTest;
using OVClassSeveralDevicesTestDefaultCore = OVClassSeveralDevicesTest;

TEST(OVClassBasicTest, smoke_SetConfigHeteroThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_HETERO, ov::enable_profiling(true)));
}

TEST(OVClassBasicTest, smoke_SetConfigDevicePropertiesThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.set_property("", ov::device::properties(CommonTestUtils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(ie.set_property(CommonTestUtils::DEVICE_CPU,
                                 ov::device::properties(CommonTestUtils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO,
                                 ov::device::properties(CommonTestUtils::DEVICE_CPU, ov::enable_profiling(true))),
                 ov::Exception);
    ASSERT_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO,
                                 ov::device::properties(CommonTestUtils::DEVICE_CPU, ov::num_streams(4))),
                 ov::Exception);
}

TEST_P(OVClassBasicTestP, SetConfigHeteroTargetFallbackThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities(target_device)));
}

TEST_P(OVClassBasicTestP, smoke_SetConfigHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string value;

    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities(target_device)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities));
    ASSERT_EQ(target_device, value);

    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities(target_device)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_HETERO, ov::device::priorities));
    ASSERT_EQ(target_device, value);
}


TEST(OVClassBasicTest, smoke_SetConfigAutoNoThrows) {
    ov::Core ie = createCoreWithTemplate();

    // priority config test
    ov::hint::Priority value;
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::LOW)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::LOW);
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::MEDIUM)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::MEDIUM);
    OV_ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority(ov::hint::Priority::HIGH)));
    OV_ASSERT_NO_THROW(value = ie.get_property(CommonTestUtils::DEVICE_AUTO, ov::hint::model_priority));
    EXPECT_EQ(value, ov::hint::Priority::HIGH);
}

TEST(OVClassBasicTest, smoke_SetConfigWithNoChangeToHWPluginThroughMetaPluginNoThrows) {
    ov::Core ie = createCoreWithTemplate();
    int32_t preValue = -1, curValue = -1;

    ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_CPU, {ov::num_streams(20)}));
    ASSERT_NO_THROW(curValue = ie.get_property(CommonTestUtils::DEVICE_CPU, ov::num_streams));
    EXPECT_EQ(curValue, 20);
    std::vector<std::string> metaDevices = {CommonTestUtils::DEVICE_AUTO,
                                            CommonTestUtils::DEVICE_MULTI,
                                            CommonTestUtils::DEVICE_HETERO};

    for (auto&& metaDevice : metaDevices) {
        ASSERT_NO_THROW(preValue = ie.get_property(CommonTestUtils::DEVICE_CPU, ov::num_streams));
        ASSERT_NO_THROW(
            ie.set_property(metaDevice, {ov::device::properties(CommonTestUtils::DEVICE_CPU, ov::num_streams(20))}));
        ASSERT_NO_THROW(curValue = ie.get_property(CommonTestUtils::DEVICE_CPU, ov::num_streams));
        EXPECT_EQ(curValue, preValue);
    }
    ASSERT_THROW(ie.set_property(CommonTestUtils::DEVICE_CPU,
                                 {ov::device::properties(CommonTestUtils::DEVICE_CPU, ov::num_streams(20))}),
                 ov::Exception);
    ASSERT_THROW(ie.set_property(CommonTestUtils::DEVICE_GPU,
                                 {ov::device::properties(CommonTestUtils::DEVICE_CPU, ov::num_streams(20))}),
                 ov::Exception);
    ASSERT_THROW(ie.set_property(CommonTestUtils::DEVICE_GPU,
                                 {ov::device::properties("GPU.0", ov::num_streams(20))}),
                 ov::Exception);
    ASSERT_THROW(ie.set_property("GPU.0",
                                 {ov::device::properties("GPU.0", ov::num_streams(20))}),
                 ov::Exception);
}

//
// QueryNetwork
//

TEST_P(OVClassNetworkTestP, QueryNetworkActualThrows) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.query_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device));
}

TEST_P(OVClassNetworkTestP, QueryNetworkHeteroActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ov::SupportedOpsMap res;
    OV_ASSERT_NO_THROW(
        res = ie.query_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, ov::device::priorities(target_device)));
    ASSERT_LT(0, res.size());
}

TEST_P(OVClassNetworkTestP, QueryNetworkMultiNoThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_NO_THROW(ie.query_model(actualNetwork, CommonTestUtils::DEVICE_MULTI));
}

TEST(OVClassBasicTest, smoke_GetMetricSupportedMetricsHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::string target_device = CommonTestUtils::DEVICE_HETERO;

    std::vector<ov::PropertyName> t;
    OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::supported_properties));

    std::cout << "Supported HETERO properties: " << std::endl;
    for (auto&& str : t) {
        std::cout << str << " is_mutable: " << str.is_mutable() << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

TEST_P(OVClassBasicTestP, smoke_GetMetricSupportedConfigKeysHeteroThrows) {
    ov::Core ie = createCoreWithTemplate();
    // TODO: check
    std::string real_target_device = CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device;
    ASSERT_THROW(ie.get_property(real_target_device, ov::supported_properties), ov::Exception);
}

TEST_P(OVClassGetConfigTest, GetConfigHeteroNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> configValues;
    OV_ASSERT_NO_THROW(configValues = ie.get_property(target_device, ov::supported_properties));

    for (auto&& confKey : configValues) {
        OV_ASSERT_NO_THROW(ie.get_property(target_device, confKey));
    }
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroThrow) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.get_property(CommonTestUtils::DEVICE_HETERO, "unsupported_config"), ov::Exception);
}

TEST_P(OVClassGetConfigTest_ThrowUnsupported, GetConfigHeteroWithDeviceThrow) {
    ov::Core ie = createCoreWithTemplate();

    ASSERT_THROW(ie.get_property(CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device,
                                   ov::device::priorities),
                 ov::Exception);
}

//
// QueryNetwork with HETERO on particular device
//
TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        OV_ASSERT_NO_THROW(ie.query_model(actualNetwork,
                                          CommonTestUtils::DEVICE_HETERO,
                                          ov::device::priorities(target_device + "." + deviceIDs[0], target_device)));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkHETEROWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.query_model(actualNetwork,
                                    CommonTestUtils::DEVICE_HETERO,
                                    ov::device::priorities(target_device + ".100", target_device)),
                     ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

using OVClassNetworkTestP = OVClassBaseTestP;


TEST_P(OVClassNetworkTestP, LoadNetworkMultiWithoutSettingDevicePrioritiesThrows) {
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

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDevice2NoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_HETERO, ov::device::priorities(target_device)));
}

TEST_P(OVClassNetworkTestP, LoadNetworkActualHeteroDeviceUsingDevicePropertiesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork,
        CommonTestUtils::DEVICE_HETERO,
        ov::device::priorities(target_device),
        ov::device::properties(target_device,
            ov::enable_profiling(true))));
}

TEST_P(OVClassSeveralDevicesTestLoadNetwork, LoadNetworkActualSeveralDevicesNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsAvailableDevices(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
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

TEST_P(OVClassSeveralDevicesTestQueryNetwork, QueryNetworkActualSeveralDevicesNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
    if (!supportsAvailableDevices(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
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

//
// LoadNetwork with HETERO on particular device
//
TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        std::string heteroDevice =
                CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device + "." + deviceIDs[0] + "," + target_device;
        OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, heteroDevice));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}


TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                      "HETERO",
                                       ov::device::priorities(target_device + ".100", CommonTestUtils::DEVICE_CPU)),
                     ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROAndDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.compile_model(actualNetwork,
                                      CommonTestUtils::DEVICE_HETERO,
                                      ov::device::priorities(target_device, CommonTestUtils::DEVICE_CPU),
                                      ov::device::id("110")),
                     ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

//
// LoadNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, LoadNetworkHETEROwithMULTINoThrow) {
    ov::Core ie = createCoreWithTemplate();
    if (supportsDeviceID(ie, target_device) && supportsAvailableDevices(ie, target_device)) {
        std::string devices;
        auto availableDevices = ie.get_property(target_device, ov::available_devices);
        for (auto&& device : availableDevices) {
            devices += target_device + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        OV_ASSERT_NO_THROW(
            ie.compile_model(actualNetwork,
                             CommonTestUtils::DEVICE_HETERO,
                             ov::device::properties(CommonTestUtils::DEVICE_MULTI,
                                               ov::device::priorities(devices)),
                             ov::device::properties(CommonTestUtils::DEVICE_HETERO,
                                               ov::device::priorities(CommonTestUtils::DEVICE_MULTI, target_device))));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, LoadNetworkMULTIwithHETERONoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device) && supportsAvailableDevices(ie, target_device)) {
        std::string devices;
        auto availableDevices = ie.get_property(target_device, ov::available_devices);
        for (auto&& device : availableDevices) {
            devices += CommonTestUtils::DEVICE_HETERO + std::string(".") + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        OV_ASSERT_NO_THROW(ie.compile_model(
            actualNetwork,
            CommonTestUtils::DEVICE_MULTI,
            ov::device::properties(CommonTestUtils::DEVICE_MULTI, ov::device::priorities(devices)),
            ov::device::properties(CommonTestUtils::DEVICE_HETERO, ov::device::priorities(target_device, target_device))));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

//
// QueryNetwork with HETERO on MULTI combinations particular device
//

TEST_P(OVClassLoadNetworkTest, QueryNetworkHETEROWithMULTINoThrow_V10) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device) && supportsAvailableDevices(ie, target_device)) {
        std::string devices;
        auto availableDevices = ie.get_property(target_device, ov::available_devices);
        for (auto&& device : availableDevices) {
            devices += target_device + '.' + device;
            if (&device != &(availableDevices.back())) {
                devices += ',';
            }
        }
        auto function = multinputNetwork;
        ASSERT_NE(nullptr, function);
        std::unordered_set<std::string> expectedLayers;
        for (auto&& node : function->get_ops()) {
            expectedLayers.emplace(node->get_friendly_name());
        }
        ov::SupportedOpsMap result;
        std::string hetero_device_priorities(CommonTestUtils::DEVICE_MULTI + std::string(",") + target_device);
        OV_ASSERT_NO_THROW(result = ie.query_model(
                            multinputNetwork,
                            CommonTestUtils::DEVICE_HETERO,
                            ov::device::properties(CommonTestUtils::DEVICE_MULTI,
                                                   ov::device::priorities(devices)),
                            ov::device::properties(CommonTestUtils::DEVICE_HETERO,
                                                   ov::device::priorities(CommonTestUtils::DEVICE_MULTI,
                                                                          target_device))));

        std::unordered_set<std::string> actualLayers;
        for (auto&& layer : result) {
            actualLayers.emplace(layer.first);
        }
        ASSERT_EQ(expectedLayers, actualLayers);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassLoadNetworkTest, QueryNetworkMULTIWithHETERONoThrow_V10) {
    ov::Core ie = createCoreWithTemplate();

    if (!supportsDeviceID(ie, target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
    if (!supportsAvailableDevices(ie, target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
    }
    std::string devices;
    auto availableDevices = ie.get_property(target_device, ov::available_devices);
    for (auto&& device : availableDevices) {
        devices += std::string(CommonTestUtils::DEVICE_HETERO) + "." + device;
        if (&device != &(availableDevices.back())) {
            devices += ',';
        }
    }
    auto function = multinputNetwork;
    ASSERT_NE(nullptr, function);
    std::unordered_set<std::string> expectedLayers;
    for (auto&& node : function->get_ops()) {
        expectedLayers.emplace(node->get_friendly_name());
    }
    ov::SupportedOpsMap result;
    OV_ASSERT_NO_THROW(result = ie.query_model(multinputNetwork,
                                                CommonTestUtils::DEVICE_MULTI,
                                                ov::device::properties(CommonTestUtils::DEVICE_MULTI,
                                                                ov::device::priorities(devices)),
                                                ov::device::properties(CommonTestUtils::DEVICE_HETERO,
                                                                ov::device::priorities(target_device, target_device))));

    std::unordered_set<std::string> actualLayers;
    for (auto&& layer : result) {
        actualLayers.emplace(layer.first);
    }
    ASSERT_EQ(expectedLayers, actualLayers);
}

// TODO: Enable this test with pre-processing
TEST_P(OVClassLoadNetworkAfterCoreRecreateTest, LoadAfterRecreateCoresAndPlugins) {
    ov::Core ie = createCoreWithTemplate();
    {
        auto versions = ie.get_versions(std::string(CommonTestUtils::DEVICE_MULTI) + ":" + target_device + "," +
                                        CommonTestUtils::DEVICE_CPU);
        ASSERT_EQ(3, versions.size());
    }
    ov::AnyMap config;
    if (target_device == CommonTestUtils::DEVICE_CPU) {
        config.insert(ov::enable_profiling(true));
    }
    // OV_ASSERT_NO_THROW({
    //     ov::Core ie = createCoreWithTemplate();
    //     std::string name = actualNetwork.getInputsInfo().begin()->first;
    //     actualNetwork.getInputsInfo().at(name)->setPrecision(Precision::U8);
    //     auto executableNetwork = ie.compile_model(actualNetwork, target_device, config);
    // });
};


TEST_P(OVClassSeveralDevicesTestDefaultCore, DefaultCoreSeveralDevicesNoThrow) {
    ov::Core ie;

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    if (!supportsDeviceID(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
    if (!supportsAvailableDevices(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect Device ID" << std::endl;

    for (size_t i = 0; i < target_devices.size(); ++i) {
        OV_ASSERT_NO_THROW(ie.set_property(target_devices[i], ov::enable_profiling(true)));
    }
    bool res;
    for (size_t i = 0; i < target_devices.size(); ++i) {
        OV_ASSERT_NO_THROW(res = ie.get_property(target_devices[i], ov::enable_profiling));
        ASSERT_TRUE(res);
    }
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
