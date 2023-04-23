// Copyright (C) 2018-2023 Intel Corporation
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

class OVEmptyPropertiesTests : public testing::WithParamInterface<std::string>, public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);

    void SetUp() override;
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
using OVSetSupportPropCompileModelWithoutConfigTests = OVPropertiesTests;
using OVSetUnsupportPropCompileModelWithoutConfigTests = OVPropertiesTests;

using CompileModelPropertiesParams = std::tuple<std::string, AnyMap, AnyMap>;
class OVSetPropComplieModleGetPropTests : public testing::WithParamInterface<CompileModelPropertiesParams>,
                                          public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileModelPropertiesParams> obj);

    void SetUp() override;

    AnyMap compileModelProperties;
};

using OVSetPropCompileModelWithIncorrectPropTests = OVSetPropComplieModleGetPropTests;

class OVPropertiesTestsWithComplieModelProps : public testing::WithParamInterface<PropertiesParams>,
                                               public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj);

    void SetUp() override;

    void TearDown() override;

    AnyMap compileModelProperties;

    static std::vector<ov::AnyMap> getPropertiesValues();
    static std::vector<ov::AnyMap> getModelDependcePropertiesValues();
};

using OVCheckChangePropComplieModleGetPropTests = OVPropertiesTestsWithComplieModelProps;
using OVCheckChangePropComplieModleGetPropTests_DEVICE_ID = OVPropertiesTestsWithComplieModelProps;
using OVCheckChangePropComplieModleGetPropTests_ModelDependceProps = OVPropertiesTestsWithComplieModelProps;

using OvPropertiesParams =
    std::tuple<std::string,                        // device name
               std::pair<ov::AnyMap, std::string>  // device and expect execution device configuration
               >;
class OVCompileModelGetExecutionDeviceTests : public testing::WithParamInterface<OvPropertiesParams>,
                                              public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<OvPropertiesParams> obj);

    void SetUp() override;

    AnyMap compileModelProperties;

    std::string expectedDeviceName;
};

using OVClassExecutableNetworkGetMetricTest_EXEC_DEVICES = OVCompileModelGetExecutionDeviceTests;

class OVClassSetDefaultDeviceIDPropTest : public OVPluginTestBase,
                                          public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string deviceID;

public:
    void SetUp() override {
        std::tie(target_device, deviceID) = GetParam();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
    }
};

using OVClassCompileModelWithCorrectPropertiesTest = OVClassSetDevicePriorityConfigPropsTest;
using OVClassCompileModelWithCondidateDeviceListContainedMetaPluginTest = OVClassSetDevicePriorityConfigPropsTest;
using OVClassCompileModelReturnDefaultHintTest = OVClassSetDevicePriorityConfigPropsTest;
using OVClassCompileModelDoNotReturnDefaultHintTest = OVClassSetDevicePriorityConfigPropsTest;
using OVClassCompileModelAndCheckSecondaryPropertiesTest = OVClassSetDevicePriorityConfigPropsTest;

using OVGetConfigTest = OVClassBaseTestP;
using OVSpecificDeviceGetConfigTest = OVClassBaseTestP;
using OVGetConfigTest_ThrowUnsupported = OVClassBaseTestP;
using OVGetAvailableDevicesPropsTest = OVClassBaseTestP;
using OVGetMetricPropsTest = OVClassBaseTestP;
using OVSetEnableCpuPinningHintConfigTest = OVClassBaseTestP;
using OVSetSchedulingCoreTypeHintConfigTest = OVClassBaseTestP;
using OVSetEnableHyperThreadingHintConfigTest = OVClassBaseTestP;

using OVSetModelPriorityConfigTest = OVClassBaseTestP;
using OVSetExecutionModeHintConfigTest = OVClassBaseTestP;
using OVSetLogLevelConfigTest = OVClassBaseTestP;
using OVSpecificDeviceTestSetConfig = OVClassBaseTestP;

class OVClassBasicPropsTestP : public OVPluginTestBase,
                               public ::testing::WithParamInterface<std::pair<std::string, std::string>> {
protected:
    std::string deviceName;
    std::string pluginName;

public:
    void SetUp() override {
        std::tie(pluginName, target_device) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        pluginName += IE_BUILD_POSTFIX;
        if (pluginName == (std::string("openvino_template_plugin") + IE_BUILD_POSTFIX)) {
            pluginName = ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(), pluginName);
        }
    }
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
