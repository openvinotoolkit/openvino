// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

const std::vector<ov::AnyMap> inproperties = {
        {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVClassCompiledModelPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::ValuesIn(inproperties)),
                        OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
        {ov::enable_profiling(false)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVClassCompiledModelPropertiesDefaultTests,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::target_device),
                ::testing::ValuesIn(default_properties)),
        OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVCompiledModelPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::target_device),
                         OVCompiledModelPropertiesDefaultSupportedTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVClassCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::target_device),
                ::testing::ValuesIn(default_properties)),
        OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassCompiledModelEmptyPropertiesTests,
        ::testing::Values(ov::test::utils::target_device));

// OV Class Load network

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCompiledModelIncorrectDevice,
        ::testing::Values(ov::test::utils::target_device));

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory,
                         OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values(ov::test::utils::target_device),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));

} // namespace
