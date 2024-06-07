// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
    {ov::enable_profiling(false)},
    {ov::device::id(0)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(default_properties)),
                         OVPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

const std::vector<ov::AnyMap> properties = {
    {ov::enable_profiling(true)},
    {ov::device::id(0)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(properties)),
                         OVPropertiesTests::getTestCaseName);

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest,
                         OVGetMetricPropsTest,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
    smoke_OVCheckGetSupportedROMetricsPropsTests,
    OVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

//
// OV Class GetConfig
//
INSTANTIATE_TEST_SUITE_P(smoke_OVBasicPropertiesTestsP,
                         OVBasicPropertiesTestsP,
                         ::testing::Values(std::make_pair("openvino_template_plugin",
                                                          ov::test::utils::DEVICE_TEMPLATE)));
}  // namespace
