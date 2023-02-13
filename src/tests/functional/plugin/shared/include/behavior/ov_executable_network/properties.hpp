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

class OVCompiledModelPropertiesBase : public OVCompiledNetworkTestBase {
public:
    std::shared_ptr<Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> model;
    AnyMap properties;
};

class OVCompiledModelEmptyPropertiesTests : public testing::WithParamInterface<std::string>,
                                            public OVCompiledModelPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
};

using PropertiesParams = std::tuple<std::string, AnyMap>;

class OVCompiledModelPropertiesTests : public testing::WithParamInterface<PropertiesParams>,
                                       public OVCompiledModelPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj);
    void SetUp() override;
    void TearDown() override;
};

using OVCompiledModelPropertiesIncorrectTests = OVCompiledModelPropertiesTests;
using OVCompiledModelPropertiesDefaultTests = OVCompiledModelPropertiesTests;

using DevicePriorityParams = std::tuple<
        std::string,            // Device name
        ov::AnyMap              // Configuration key and its default value
>;
class OVClassSetDevicePriorityConfigTest : public OVPluginTestBase,
                                           public ::testing::WithParamInterface<DevicePriorityParams> {
protected:
    std::string deviceName;
    ov::AnyMap configuration;
    std::shared_ptr<ngraph::Function> actualNetwork;

public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    }
};

using OVClassLoadNetworkWithCorrectPropertiesTest = OVClassSetDevicePriorityConfigTest;

using OVClassLoadNetWorkReturnDefaultHintTest = OVClassSetDevicePriorityConfigTest;
using OVClassLoadNetWorkDoNotReturnDefaultHintTest = OVClassSetDevicePriorityConfigTest;
using OVClassLoadNetworkAndCheckSecondaryPropertiesTest = OVClassSetDevicePriorityConfigTest;

using OVClassLoadNetworkWithCondidateDeviceListContainedMetaPluginTest = OVClassSetDevicePriorityConfigTest;

}  // namespace behavior
}  // namespace test
}  // namespace ov
