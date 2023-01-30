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

TEST_P(OVClassExecutableNetworkGetMetricTest_DEVICE_PROPERTIES, GetMetricWithDevicePropertiesNoThrow) {
    ov::Core core = createCoreWithTemplate();
    auto compiled_model = core.compile_model(simpleNetwork, target_device, configuration);
    int32_t expected_value = configuration[device_name].as<ov::AnyMap>()[ov::num_streams.name()].as<int32_t>();
    int32_t actual_value = -1;
    ASSERT_NO_THROW(actual_value = compiled_model.get_property(ov::device::properties(device_name, ov::num_streams)));
    ASSERT_EQ(expected_value, actual_value);
}

TEST_P(OVClassExecutableNetworkGetMetricTestUnsupportConfigThrow_DEVICE_PROPERTIES,
       GetMetricWithDevicePropertiesThrow) {
    ov::Core ie = createCoreWithTemplate();
    auto compiled_model = ie.compile_model(simpleNetwork, target_device, configuration);
    // throw exception when getting unsupported property through device's executable network via this API
    ASSERT_THROW(compiled_model.get_property(ov::device::properties(device_name, ov::device::priorities)),
                 ov::Exception);
}

TEST_P(OVClassExecutableNetworkGetMetricTestInvalidDeviceThrow_DEVICE_PROPERTIES, GetMetricWithDevicePropertiesThrow) {
    ov::Core core = createCoreWithTemplate();
    auto compiled_model = core.compile_model(simpleNetwork, target_device, configuration);
    // executable network is not found in meta plugin
    ASSERT_THROW(compiled_model.get_property(ov::device::properties(device_name, ov::num_streams)), ov::Exception);
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
