// Copyright (C) 2018-2023 Intel Corporation
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

const std::vector<ov::AnyMap> auto_batch_inproperties = {
        {{ov::auto_batch_timeout(-1)}},
};

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::ValuesIn(inproperties)),
                        OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_AutoBatch, OVPropertiesIncorrectTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                ::testing::ValuesIn(auto_batch_inproperties)),
                        OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> default_properties = {
        {ov::enable_profiling(true)},
        {ov::device::id("0")},
};

const std::vector<ov::AnyMap> auto_batch_properties = {
        {},
        {{CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "1"}},
        {{ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVPropertiesTests,
        ::testing::Combine(
                ::testing::ValuesIn(return_all_possible_device_combination(false)),
                ::testing::ValuesIn(default_properties)),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin_AutoBatch, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                ::testing::ValuesIn(ov::test::conformance::generate_ov_configs(CommonTestUtils::DEVICE_BATCH, auto_batch_properties))),
        OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckChangePropComplieModleGetPropTests,
        ::testing::Combine(
                        ::testing::ValuesIn(return_all_possible_device_combination()),
                        ::testing::ValuesIn(OVCheckChangePropComplieModleGetPropTests::getPropertiesValues())),
        OVCheckChangePropComplieModleGetPropTests::getTestCaseName);

const std::vector<ov::AnyMap> device_properties = {
        {ov::device::id("0")},
};

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckChangePropComplieModleGetPropTests_DEVICE_ID,
        ::testing::Combine(
                ::testing::ValuesIn(return_all_possible_device_combination()),
                ::testing::ValuesIn(device_properties)),
        OVCheckChangePropComplieModleGetPropTests_DEVICE_ID::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVCheckChangePropComplieModleGetPropTests_ModelDependceProps,
        ::testing::Combine(
                ::testing::ValuesIn(return_all_possible_device_combination()),
                ::testing::ValuesIn(OVCheckChangePropComplieModleGetPropTests::getModelDependcePropertiesValues())),
        OVCheckChangePropComplieModleGetPropTests_ModelDependceProps::getTestCaseName);
} // namespace
