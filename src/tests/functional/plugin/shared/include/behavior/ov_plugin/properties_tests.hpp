// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace test {
namespace behavior {

#define OV_ASSERT_PROPERTY_SUPPORTED(property_key)                                  \
    {                                                                               \
        auto properties = ie.get_property(target_device, ov::supported_properties); \
        auto it = std::find(properties.begin(), properties.end(), property_key);    \
        ASSERT_NE(properties.end(), it);                                            \
    }

class OVPropertiesBase : public OVPluginTestBase {
public:
    std::shared_ptr<Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> model;
    AnyMap properties;
};

using PropertiesParams = std::tuple<std::string, AnyMap>;

class OVPropertiesTests : public testing::WithParamInterface<PropertiesParams>, public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj);

    void SetUp() override;

    void TearDown() override;
};

using OVPropertiesIncorrectTests = OVPropertiesTests;
using OVPropertiesDefaultTests = OVPropertiesTests;
using OVPropertiesDefaultSupportedTests = OVClassBaseTestP;
using OVSetUnsupportPropCompileModelWithoutConfigTests = OVPropertiesTests;

using CompileModelPropertiesParams = std::tuple<std::string, AnyMap, AnyMap>;
class OVSetPropComplieModleGetPropTests : public testing::WithParamInterface<CompileModelPropertiesParams>,
                                          public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileModelPropertiesParams> obj);

    void SetUp() override;

    AnyMap compileModelProperties;
};

class OVPropertiesTestsWithCompileModelProps : public testing::WithParamInterface<PropertiesParams>,
                                               public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj);

    void SetUp() override;

    void TearDown() override;

    AnyMap compileModelProperties;

    static std::vector<ov::AnyMap> getROMandatoryProperties(bool is_sw_device = false);
    static std::vector<ov::AnyMap> getROOptionalProperties(bool is_sw_device = false);
    static std::vector<ov::AnyMap> configureProperties(std::vector<std::string> props);

    static std::vector<ov::AnyMap> getRWMandatoryPropertiesValues(const std::vector<std::string>& props = {}, bool is_sw_device = false);
    static std::vector<ov::AnyMap> getWrongRWMandatoryPropertiesValues(const std::vector<std::string>& props = {}, bool is_sw_device = false);
    static std::vector<ov::AnyMap> getRWOptionalPropertiesValues(const std::vector<std::string>& props = {}, bool is_sw_device = false);
    static std::vector<ov::AnyMap> getWrongRWOptionalPropertiesValues(const std::vector<std::string>& props = {}, bool is_sw_device = false);

    static std::vector<ov::AnyMap> getModelDependcePropertiesValues();
};

using OVCheckSetSupportedRWMetricsPropsTests = OVPropertiesTestsWithCompileModelProps;
using OVCheckSetIncorrectRWMetricsPropsTests = OVPropertiesTestsWithCompileModelProps;
using OVCheckGetSupportedROMetricsPropsTests = OVPropertiesTestsWithCompileModelProps;
using OVCheckChangePropComplieModleGetPropTests_DEVICE_ID = OVPropertiesTestsWithCompileModelProps;
using OVCheckChangePropComplieModleGetPropTests_InferencePrecision = OVPropertiesTestsWithCompileModelProps;
using OVCheckMetricsPropsTests_ModelDependceProps = OVPropertiesTestsWithCompileModelProps;

class OVClassSetDefaultDeviceIDPropTest : public OVPluginTestBase,
                                          public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string deviceID;

public:
    void SetUp() override {
        std::tie(target_device, deviceID) = GetParam();
        ov::Core ie = ov::test::utils::create_core();
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
            if (target_device == "GPU" && deviceID == "1") {
                GTEST_SKIP() << "Skip this test, because gpu with number 1 not available and test check properties of GPU.1";
            }
        }
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
    }
};


using OVClassCompileModelWithCondidateDeviceListContainedMetaPluginTest = OVClassSetDevicePriorityConfigPropsTest;
using OVClassCompileModelReturnDefaultHintTest = OVClassSetDevicePriorityConfigPropsTest;
using OVClassCompileModelDoNotReturnDefaultHintTest = OVClassSetDevicePriorityConfigPropsTest;
using OVClassCompileModelAndCheckSecondaryPropertiesTest = OVClassSetDevicePriorityConfigPropsTest;
using OVGetConfigTest = OVClassBaseTestP;

class OVSpecificDeviceSetConfigTest : public OVClassBaseTestP {
public:
    void SetUp() override {
        target_device = GetParam();
        ov::Core ie = ov::test::utils::create_core();
        std::string clear_target_device, deviceID;
        auto pos = target_device.find('.');
        if (pos != std::string::npos) {
            clear_target_device = target_device.substr(0, pos);
            deviceID =  target_device.substr(pos + 1,  target_device.size());
        }
        // sw plugins are not requested to support `ov::available_devices`, `ov::device::id` and `ov::num_streams` property
        auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
        if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
            if (target_device == "GPU.1") {
                GTEST_SKIP() << "Skip this test, because gpu with number 1 not available and test check properties of GPU.1";
            }
        }
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
    }
};

using OVSpecificDeviceGetConfigTest = OVSpecificDeviceSetConfigTest;
using OVGetAvailableDevicesPropsTest = OVClassBaseTestP;
using OVGetMetricPropsTest = OVClassBaseTestP;
using OVGetMetricPropsOptionalTest = OVClassBaseTestP;
using OVSetEnableHyperThreadingHintConfigTest = OVClassBaseTestP;

using OVSpecificDeviceTestSetConfig = OVSpecificDeviceSetConfigTest;

class OVBasicPropertiesTestsP : public OVPluginTestBase,
                               public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string pluginName;
public:
    void SetUp() override {
        std::tie(pluginName, target_device) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        pluginName += OV_BUILD_POSTFIX;
        if (pluginName == (std::string("openvino_template_plugin") + OV_BUILD_POSTFIX)) {
            pluginName = ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(), pluginName);
        }
    }
};

using OVClassSeveralDevicesTestDefaultCore = OVClassSeveralDevicesTests;

}  // namespace behavior
}  // namespace test
}  // namespace ov
