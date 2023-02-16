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

using OVCompiledModelClassBaseTest = OVCompiledModelClassBaseTestP;
using CompiledModelImportExportTestP = OVCompiledModelClassBaseTestP;
using CompiledModelGetMetricTest_SUPPORTED_CONFIG_KEYS = OVCompiledModelClassBaseTestP;
using CompiledModelGetMetricTest_SUPPORTED_METRICS = OVCompiledModelClassBaseTestP;
using CompiledModelGetMetricTest_NETWORK_NAME = OVCompiledModelClassBaseTestP;
using CompiledModelGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = OVCompiledModelClassBaseTestP;
using CompiledModelGetMetricTest_ThrowsUnsupported = OVCompiledModelClassBaseTestP;
using CompiledModelPropertyTest = OVCompiledModelClassBaseTestP;
using CompiledModelSetConfigTest = OVCompiledModelClassBaseTestP;

class CompiledModelGetMetricTestForSpecificConfig :
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

using CompiledModelSupportedConfigTest = CompiledModelGetMetricTestForSpecificConfig;
using CompiledModelUnsupportedConfigTest = CompiledModelGetMetricTestForSpecificConfig;

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
        heteroDeviceName = CommonTestUtils::DEVICE_HETERO + std::string(":") + target_device;
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
