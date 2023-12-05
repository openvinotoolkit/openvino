// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"

#include "ie_system_conf.h"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> auto_batch_inproperties = {
    {ov::num_streams(-100)},
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inproperties)),
                         OVCompiledModelPropertiesIncorrectTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_batch_properties = {
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
     {ov::auto_batch_timeout(1)}},
    {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
     {ov::auto_batch_timeout(10)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_properties)),
                         OVCompiledModelPropertiesTests::getTestCaseName);

}  // namespace
