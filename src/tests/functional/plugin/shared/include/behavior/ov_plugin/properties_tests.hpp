// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVPropertiesBase : public OVPluginTestBase {
public:
    std::shared_ptr<Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> model;
    AnyMap properties;
};

class OVEmptyPropertiesTests : public testing::WithParamInterface<std::string>,
                               public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);

    void SetUp() override;
};

using PropertiesParams = std::tuple<std::string, AnyMap>;

class OVPropertiesTests : public testing::WithParamInterface<PropertiesParams>,
                          public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj);

    void SetUp() override;

    void TearDown() override;
};

using OVPropertiesIncorrectTests = OVPropertiesTests;
using OVGetPropertiesIncorrectTests = OVPropertiesTests;
using OVPropertiesDefaultTests = OVPropertiesTests;
using OVSetSupportPropComplieModleWithoutConfigTests = OVPropertiesTests;
using OVSetUnsupportPropComplieModleWithoutConfigTests = OVPropertiesTests;

using CompileModelPropertiesParams = std::tuple<std::string, AnyMap, AnyMap>;
class OVSetPropComplieModleGetPropTests : public testing::WithParamInterface<CompileModelPropertiesParams>,
                                          public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<CompileModelPropertiesParams> obj);

    void SetUp() override;

    AnyMap compileModelProperties;
};

using OVSetPropComplieModleWihtIncorrectPropTests = OVSetPropComplieModleGetPropTests;

class OVPropertiesTestsWithComplieModelProps : public testing::WithParamInterface<PropertiesParams>,
                                               public OVPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj);

    void SetUp() override;

    void TearDown() override;

    AnyMap compileModelProperties;

    static std::vector<ov::AnyMap> getAllROPropertiesValues();
    static std::vector<ov::AnyMap> getRWPropertiesValues(std::vector<std::string> param_properties = {});
    static std::vector<ov::AnyMap> getModelDependcePropertiesValues();
};

using OVCheckChangePropComplieModleGetPropTestsRO = OVPropertiesTestsWithComplieModelProps;
using OVCheckChangePropComplieModleGetPropTestsRW = OVPropertiesTestsWithComplieModelProps;
using OVCheckChangePropComplieModleGetPropTests_DEVICE_ID = OVPropertiesTestsWithComplieModelProps;
using OVCheckChangePropComplieModleGetPropTests_ModelDependceProps = OVPropertiesTestsWithComplieModelProps;

using OvPropertiesParams = std::tuple<
        std::string,                          // device name
        std::pair<ov::AnyMap, std::string>    // device and expect execution device configuration
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

class OVClassSetDefaultDeviceIDTest : public OVPluginTestBase,
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

using OVClassGetConfigTest = OVClassBaseTestP;
using OVClassSetGlobalConfigTest = OVClassBaseTestP;
using OVClassGetAvailableDevices = OVClassBaseTestP;
using OVClassSpecificDeviceTestSetConfig = OVClassBaseTestP;
using OVClassSpecificDeviceTestGetConfig = OVClassBaseTestP;
using OVClassGetMetricTest_RANGE_FOR_STREAMS = OVClassBaseTestP;
using OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS = OVClassBaseTestP;
using OVClassGetMetricTest_FULL_DEVICE_NAME_with_DEVICE_ID = OVClassBaseTestP;

}  // namespace behavior
}  // namespace test
}  // namespace ov
