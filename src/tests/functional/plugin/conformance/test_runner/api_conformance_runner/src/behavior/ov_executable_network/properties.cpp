// Copyright (C) 2018-2023 Intel Corporation
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

const std::vector<ov::AnyMap> auto_batch_inproperties = {};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVClassCompiledModelPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::conformance::return_all_possible_device_combination()),
                                ::testing::ValuesIn(inproperties)),
                        OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_AutoBatch, OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(
                                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(generate_ov_configs(ov::test::utils::DEVICE_BATCH, inproperties))),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);


const std::vector<ov::AnyMap> default_properties = {
        {ov::enable_profiling(false)}
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVClassCompiledModelPropertiesDefaultTests,
        ::testing::Combine(
                ::testing::ValuesIn(return_all_possible_device_combination()),
                ::testing::ValuesIn(default_properties)),
        OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVCompiledModelPropertiesDefaultSupportedTests,
                         ::testing::ValuesIn(return_all_possible_device_combination()),
                         OVCompiledModelPropertiesDefaultSupportedTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_batch_properties = {
        {{CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "1"}},
        {{ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory, OVClassCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::conformance::return_all_possible_device_combination()),
                ::testing::ValuesIn(default_properties)),
        OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_AutoBatch, OVClassCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::Values(ov::test::utils::DEVICE_BATCH),
                ::testing::ValuesIn(ov::test::conformance::generate_ov_configs(ov::test::utils::DEVICE_BATCH, auto_batch_properties))),
        OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassCompiledModelEmptyPropertiesTests,
        ::testing::ValuesIn(ov::test::conformance::return_all_possible_device_combination()));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory, OVCompiledModelIncorrectDevice,
        ::testing::Values(targetDevice));

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_mandatory,
                         OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::ValuesIn(return_all_possible_device_combination()),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));

} // namespace
