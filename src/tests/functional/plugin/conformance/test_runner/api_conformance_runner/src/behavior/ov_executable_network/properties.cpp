// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"
#include "behavior/ov_executable_network/hetero_properties.hpp"
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

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassCompiledModelProperties_SupportedProperties,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassCompiledModelGetIncorrectProperties,
        ::testing::ValuesIn(return_all_possible_device_combination()));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVCompiledModelGetSupportedPropertiesTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(ov_compiled_model,
                         OVClassCompiledModelUnsupportedConfigTest,
                         ::testing::Combine(::testing::ValuesIn(return_all_possible_device_combination()),
                                            ::testing::Values(std::make_pair("unsupported_config", "some_value"))));

////
//// Hetero Executable Network GetMetric
////

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(targetDevice));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(targetDevice));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values(targetDevice));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values(targetDevice));
} // namespace
