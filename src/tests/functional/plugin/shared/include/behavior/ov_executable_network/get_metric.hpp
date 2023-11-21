// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <base/ov_behavior_test_utils.hpp>

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

using PriorityParams = std::tuple<
        std::string,            // Device name
        ov::AnyMap              // device priority Configuration key
>;
class OVClassExecutableNetworkGetMetricTest_Priority : public ::testing::WithParamInterface<PriorityParams>,
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
        simpleNetwork = ngraph::builder::subgraph::makeSingleConv();
    }
};
using OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY = OVClassExecutableNetworkGetMetricTest_Priority;
using OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY = OVClassExecutableNetworkGetMetricTest_Priority;

using OVCompiledModelClassBaseTest = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkImportExportTestP = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_NETWORK_NAME = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetConfigTest = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkSetConfigTest = OVCompiledModelClassBaseTestP;
using OVClassExecutableNetworkGetConfigTest = OVCompiledModelClassBaseTestP;

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
using OVClassExecutableNetworkUnsupportedConfigTest = OVClassExecutableNetworkGetMetricTestForSpecificConfig;

//
// Hetero Executable network case
//
class OVClassHeteroExecutableNetworkGetMetricTest :
        public OVClassNetworkTest,
        public ::testing::WithParamInterface<std::string>,
        public OVCompiledNetworkTestBase {
protected:
    std::string heteroDeviceName;
    void SetCpuAffinity(ov::Core& core, std::vector<std::string>& expectedTargets);

public:
    void SetUp() override {
        target_device = GetParam();
        heteroDeviceName = ov::test::utils::DEVICE_HETERO + std::string(":") + target_device;
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        OVClassNetworkTest::SetUp();
    }
};
using OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK = OVClassHeteroExecutableNetworkGetMetricTest;
using OVClassHeteroExecutableNetworkGetMetricTest_EXEC_DEVICES = OVClassHeteroExecutableNetworkGetMetricTest;

}  // namespace behavior
}  // namespace test
}  // namespace ov
