// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> auto_batch_inproperties = {
    {ov::num_streams(-100)},
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_batch_properties = {
    {{ov::device::priorities.name(), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"}},
    {{ov::device::priorities.name(), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
     {ov::auto_batch_timeout(1)}},
    {{ov::device::priorities.name(), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
     {ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

}  // namespace
