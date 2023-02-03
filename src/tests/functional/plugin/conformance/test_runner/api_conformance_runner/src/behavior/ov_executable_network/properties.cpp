// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

const std::vector<ov::AnyMap> inproperties = {
        {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> auto_batch_inproperties = {
        {{ov::auto_batch_timeout(-1)}},
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model, OVCompiledModelPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::conformance::return_all_possible_device_combination()),
                                ::testing::ValuesIn(inproperties)),
                        OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_AutoBatch, OVCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(generate_ov_configs(CommonTestUtils::DEVICE_BATCH, auto_batch_inproperties))),
                         OVCompiledModelPropertiesIncorrectTests::getTestCaseName);


const std::vector<ov::AnyMap> default_properties = {
        {ov::enable_profiling(false)},
        {ov::device::id("0")},
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model, OVCompiledModelPropertiesDefaultTests,
        ::testing::Combine(
                ::testing::ValuesIn(return_all_possible_device_combination()),
                ::testing::ValuesIn(default_properties)),
        OVCompiledModelPropertiesDefaultTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_batch_properties = {
        {{CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "1"}},
        {{ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model, OVCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::conformance::return_all_possible_device_combination()),
                ::testing::ValuesIn(default_properties)),
        OVCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_compiled_model_AutoBatch, OVCompiledModelPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                ::testing::ValuesIn(ov::test::conformance::generate_ov_configs(CommonTestUtils::DEVICE_BATCH, auto_batch_properties))),
        OVCompiledModelPropertiesTests::getTestCaseName);
} // namespace
