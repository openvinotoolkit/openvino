// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

const std::vector<ov::AnyMap> inproperties = {
        {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::ValuesIn(inproperties)),
                        OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
        {},
        {ov::enable_profiling(false)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
};

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::target_device),
                ::testing::ValuesIn(default_properties)),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVPropertiesDefaultTests,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::target_device),
                ::testing::ValuesIn(default_properties)),
        OVPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCheckGetSupportedROMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(ov::test::utils::target_device),
                        ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::getROMandatoryProperties(
                                sw_plugin_in_target_device(ov::test::utils::target_device)))),
        OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckGetSupportedROMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(ov::test::utils::target_device),
                        ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::getROOptionalProperties(
                                sw_plugin_in_target_device(ov::test::utils::target_device)))),
        OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCheckSetSupportedRWMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(ov::test::utils::target_device),
                        ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWMandatoryPropertiesValues(
                                {}, sw_plugin_in_target_device(ov::test::utils::target_device)))),
        OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckSetSupportedRWMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(ov::test::utils::target_device),
                        ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWOptionalPropertiesValues(
                                {}, sw_plugin_in_target_device(ov::test::utils::target_device)))),
        OVCheckSetSupportedRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCheckSetIncorrectRWMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(ov::test::utils::target_device),
                        ::testing::ValuesIn(OVCheckSetIncorrectRWMetricsPropsTests::getWrongRWMandatoryPropertiesValues(
                                {}, sw_plugin_in_target_device(ov::test::utils::target_device)))),
        OVCheckSetIncorrectRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckSetIncorrectRWMetricsPropsTests,
        ::testing::Combine(
                        ::testing::Values(ov::test::utils::target_device),
                        ::testing::ValuesIn(OVCheckSetIncorrectRWMetricsPropsTests::getWrongRWOptionalPropertiesValues(
                                {}, sw_plugin_in_target_device(ov::test::utils::target_device)))),
        OVCheckSetIncorrectRWMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    ov_plugin,
    OVCheckChangePropComplieModleGetPropTests_DEVICE_ID,
    ::testing::Combine(::testing::Values(ov::test::utils::target_device), ::testing::Values(ov::AnyMap({}))),
    MARK_MANDATORY_PROPERTY_FOR_HW_DEVICE(OVCheckChangePropComplieModleGetPropTests_DEVICE_ID::getTestCaseName));

/* Add prefix mandatory_ to suffix (getTestCaseName) of HW plugin test cases */
INSTANTIATE_TEST_SUITE_P(
    ov_plugin,
    OVCheckChangePropComplieModleGetPropTests_InferencePrecision,
    ::testing::Combine(::testing::Values(ov::test::utils::target_device), ::testing::Values(ov::AnyMap({}))),
    MARK_MANDATORY_PROPERTY_FOR_HW_DEVICE(OVCheckChangePropComplieModleGetPropTests_InferencePrecision::getTestCaseName));

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckMetricsPropsTests_ModelDependceProps,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::target_device),
                ::testing::ValuesIn(OVCheckMetricsPropsTests_ModelDependceProps::getModelDependcePropertiesValues())),
        OVCheckMetricsPropsTests_ModelDependceProps::getTestCaseName);

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(ov_plugin,
                         OVGetMetricPropsTest,
                         ::testing::Values(ov::test::utils::target_device),
                         MARK_MANDATORY_API_FOR_HW_DEVICE_WITHOUT_PARAM());

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVGetMetricPropsOptionalTest,
        ::testing::Values(ov::test::utils::target_device));

INSTANTIATE_TEST_SUITE_P(ov_plugin,
                         OVGetAvailableDevicesPropsTest,
                         ::testing::Values(ov::test::utils::target_device),
                         MARK_MANDATORY_API_FOR_HW_DEVICE_WITHOUT_PARAM());

//
// OV Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVPropertiesDefaultSupportedTests,
        ::testing::Values(ov::test::utils::target_device));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin_mandatory, OVBasicPropertiesTestsP,
        ::testing::ValuesIn(generate_ov_pairs_plugin_name_by_device()));
} // namespace
