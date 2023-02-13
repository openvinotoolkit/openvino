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


#include <gtest/gtest.h>

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>

#endif


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

using OVClassCompiledModelProperties_SupportedProperties = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_NETWORK_NAME = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = OVCompiledModelClassBaseTestP;
using OVClassCompiledModelGetIncorrectProperties = OVCompiledModelClassBaseTestP;
using OVCompiledModelGetSupportedPropertiesTest = OVCompiledModelClassBaseTestP;

class OVClassExecutableNetworkGetMetricTestForSpecificConfig :
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

using OVClassExecutableNetworkSupportedConfigTest = OVClassExecutableNetworkGetMetricTestForSpecificConfig;
using OVClassCompiledModelUnsupportedConfigTest = OVClassExecutableNetworkGetMetricTestForSpecificConfig;

}  // namespace behavior
}  // namespace test
}  // namespace ov
