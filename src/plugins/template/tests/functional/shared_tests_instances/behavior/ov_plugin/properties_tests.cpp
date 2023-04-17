// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> hetero_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> multi_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> auto_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> auto_batch_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
    {ov::enable_profiling(true)},
    {ov::device::id(0)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(default_properties)),
                         OVPropertiesDefaultTests::getTestCaseName);

const std::vector<ov::AnyMap> properties = {
    {ov::enable_profiling(true)},
    {ov::device::id(0)},
};

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE), ov::device::id(0)},
};

const std::vector<ov::AnyMap> multi_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE), ov::device::id(0)},
};

const std::vector<ov::AnyMap> auto_batch_properties = {
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE}, {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "1"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVPropertiesTests::getTestCaseName);

const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_TEMPLATE =
    {{CommonTestUtils::DEVICE_TEMPLATE, std::make_pair(ov::AnyMap{}, "TEMPLATE.0")}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_EXEC_DEVICES,
                         ::testing::ValuesIn(GetMetricTest_ExecutionDevice_TEMPLATE),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetConfigTest,
                         OVGetConfigTest_ThrowUnsupported,
                         ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_OVGetAvailableDevicesPropsTest,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

//
// OV Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(smoke_OVGetConfigTest, OVGetConfigTest, ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassBasicPropsTestP,
                         OVClassBasicPropsTestP,
                         ::testing::Values(std::make_pair("openvino_template_plugin",
                                                          CommonTestUtils::DEVICE_TEMPLATE)));
}  // namespace
