// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
        {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
        {{"unsupported_key", "4"}}
};

const std::vector<ov::AnyMap> auto_batch_inproperties = {
        {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_BehaviorTests, OVPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE, CommonTestUtils::DEVICE_HETERO, 
                                                  CommonTestUtils::DEVICE_MULTI, CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(inproperties)),
                        OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_AutoBatch_BehaviorTests, OVPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                ::testing::ValuesIn(auto_batch_inproperties)),
                        OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
        {ov::enable_profiling(true)},
        {ov::device::id(0)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVPropertiesDefaultTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
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
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_TEMPLATE}},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_TEMPLATE},
         {CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "1"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                ::testing::ValuesIn(properties)),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Hetero_BehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                ::testing::ValuesIn(hetero_properties)),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Multi_BehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(multi_properties)),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_AutoBatch_BehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                ::testing::ValuesIn(auto_batch_properties)),
        OVPropertiesTests::getTestCaseName);


//
// OV Class GetMetric
//

const std::vector<ov::AnyMap> ro_property_template_plugin = {
        {{ov::PropertyName(ov::supported_properties.name(), ov::supported_properties.mutability), nullptr}},
        {{ov::PropertyName(ov::available_devices.name(), ov::available_devices.mutability), nullptr}},
        {{ov::PropertyName(ov::device::full_name.name(), ov::device::full_name.mutability), nullptr}},
        {{ov::PropertyName(ov::device::capabilities.name(), ov::device::capabilities.mutability), nullptr}}
};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVCheckChangePropComplieModleGetPropTestsRO, OVCheckChangePropComplieModleGetPropTestsRO,
        ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                           ::testing::ValuesIn(ro_property_template_plugin)));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

// 
// Check set of incorrect properties 
//
std::vector<ov::AnyMap> incorrect_properies = {{{"unsupported_key", "4"}}};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVGetPropertiesIncorrectTests, OVGetPropertiesIncorrectTests,
        ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                           ::testing::ValuesIn(incorrect_properies)));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetAvailableDevices, OVClassGetAvailableDevices,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));


} // namespace
