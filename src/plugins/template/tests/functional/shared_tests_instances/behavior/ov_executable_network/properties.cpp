// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> auto_batch_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE,
                                                              CommonTestUtils::DEVICE_HETERO,
                                                              CommonTestUtils::DEVICE_MULTI,
                                                              CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
    {ov::enable_profiling(true)},
    {{ov::loaded_from_cache.name(), false}},
    {ov::device::id("0")},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(default_properties)),
                         OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelPropertiesDefaultSupportedTests,
                         ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                         OVCompiledModelPropertiesDefaultSupportedTests::getTestCaseName);

const std::vector<ov::AnyMap> properties = {
    {ov::enable_profiling(true)},
    {ov::device::id("0")},
};

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE), ov::device::id("0")},
};

const std::vector<ov::AnyMap> multi_properties = {
    {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE), ov::device::id("0")},
};

const std::vector<ov::AnyMap> auto_batch_properties = {
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_TEMPLATE) + "(4)"}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_TEMPLATE) + "(4)"},
     {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "1"}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(CommonTestUtils::DEVICE_TEMPLATE) + "(4)"},
     {ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelEmptyPropertiesTests, OVClassCompiledModelEmptyPropertiesTests,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

// OV Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVCompiledModelIncorrectDevice, OVCompiledModelIncorrectDevice,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_TEMPLATE =
    {{CommonTestUtils::DEVICE_TEMPLATE, std::make_pair(ov::AnyMap{}, "TEMPLATE.0")}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::ValuesIn(GetMetricTest_ExecutionDevice_TEMPLATE));

}  // namespace
