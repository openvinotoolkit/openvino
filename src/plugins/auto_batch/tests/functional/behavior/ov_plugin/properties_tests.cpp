// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> auto_batch_inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_AutoBatch_BehaviorTests,
                         OVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_batch_properties = {
    {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}},
    {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}, {ov::auto_batch_timeout(1)}},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_AutoBatch_BehaviorTests,
                         OVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVPropertiesTests::getTestCaseName);
}  // namespace