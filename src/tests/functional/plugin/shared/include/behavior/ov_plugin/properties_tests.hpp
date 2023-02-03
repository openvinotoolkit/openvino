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
using OVPropertiesDefaultTests = OVPropertiesTests;
using OVSetSupportPropComplieModleWithoutConfigTests = OVPropertiesTests;
using OVSetUnsupportPropComplieModleWithoutConfigTests = OVPropertiesTests;
using OVCorePropertiesTest = OVPropertiesTests;

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

    static std::vector<ov::AnyMap> getPropertiesValues();
    static std::vector<ov::AnyMap> getModelDependcePropertiesValues();
};

using OVCheckChangePropComplieModleGetPropTests = OVPropertiesTestsWithComplieModelProps;
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
}  // namespace behavior
}  // namespace test
}  // namespace ov
