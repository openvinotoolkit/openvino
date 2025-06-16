// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "common_test_utils/subgraph_builders/single_conv.hpp"

namespace ov {
namespace test {
namespace behavior {

#define ASSERT_EXEC_METRIC_SUPPORTED(property)                                                \
    {                                                                                           \
        auto properties = compiled_model.get_property(ov::supported_properties);\
        auto it = std::find(properties.begin(), properties.end(), property);                        \
        ASSERT_NE(properties.end(), it);                                                           \
    }


class OVCompiledModelPropertiesBase : public OVCompiledNetworkTestBase {
public:
    std::shared_ptr<Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> model;
    AnyMap properties;
};

class OVClassCompiledModelEmptyPropertiesTests : public testing::WithParamInterface<std::string>,
                                            public OVCompiledModelPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
};

using OVCompiledModelPropertiesDefaultSupportedTests = OVClassCompiledModelEmptyPropertiesTests;

using PropertiesParams = std::tuple<std::string, AnyMap>;
class OVClassCompiledModelPropertiesTests : public testing::WithParamInterface<PropertiesParams>,
                                       public OVCompiledModelPropertiesBase {
public:
    std::shared_ptr<Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> model;
    AnyMap properties;

    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj);
    void SetUp() override;
    void TearDown() override;
};

using OVClassCompiledModelPropertiesDefaultTests = OVClassCompiledModelPropertiesTests;
using OVClassCompiledModelPropertiesIncorrectTests = OVClassCompiledModelPropertiesTests;
using OVCompiledModelIncorrectDevice = OVClassBaseTestP;
using OVClassCompileModelWithCorrectPropertiesTest = OVClassCompiledModelPropertiesTests;

class OVClassCompiledModelSetCorrectConfigTest :
        public OVClassNetworkTest,
        public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, std::string>>>,
        public OVCompiledNetworkTestBase {
protected:
    std::string configKey;
    ov::Any configValue;

public:
    void SetUp() override {
        target_device = std::get<0>(GetParam());
        std::tie(configKey, configValue) = std::get<1>(GetParam());
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
    }
};

using OVCompiledModelClassBaseTest = OVCompiledModelClassBaseTestP;
using OVClassCompiledModelGetPropertyTest = OVCompiledModelClassBaseTestP;
using OVClassCompiledModelGetConfigTest = OVCompiledModelClassBaseTestP;
using OVClassCompiledModelGetIncorrectPropertyTest = OVCompiledModelClassBaseTestP;
using OVClassCompiledModelSetIncorrectConfigTest = OVCompiledModelClassBaseTestP;

using OvPropertiesParams =
    std::tuple<std::string,                        // device name
               std::pair<ov::AnyMap, std::string>  // device and expect execution device configuration
               >;
class OVCompileModelGetExecutionDeviceTests : public testing::WithParamInterface<OvPropertiesParams>,
                                              public OVPluginTestBase {
public:
    std::shared_ptr<Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> model;
    AnyMap properties;

    static std::string getTestCaseName(testing::TestParamInfo<OvPropertiesParams> obj);
    void SetUp() override;
    AnyMap compileModelProperties;

    std::string expectedDeviceName;
};

using OVClassCompiledModelGetPropertyTest_EXEC_DEVICES = OVCompileModelGetExecutionDeviceTests;

using PriorityParams = std::tuple<
        std::string,            // Device name
        ov::AnyMap              // device priority Configuration key
>;
class OVClassCompiledModelGetPropertyTest_Priority : public ::testing::WithParamInterface<PriorityParams>,
                                                       public OVCompiledNetworkTestBase {
protected:
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> simpleNetwork;

public:
    static std::string getTestCaseName(testing::TestParamInfo<PriorityParams> obj);
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        simpleNetwork = ov::test::utils::make_single_conv();
    }
};

using OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY = OVClassCompiledModelGetPropertyTest_Priority;
using OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY = OVClassCompiledModelGetPropertyTest_Priority;

}  // namespace behavior
}  // namespace test
}  // namespace ov
