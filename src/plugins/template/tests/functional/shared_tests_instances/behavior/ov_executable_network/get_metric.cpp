// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

namespace {

//
// Executable Network GetMetric
//

std::vector<std::string> devices = {"TEMPLATE", "MULTI:TEMPLATE"};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::ValuesIn(devices));

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {{ov::hint::model_priority(ov::hint::Priority::HIGH)},
                                                           {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
                                                           {ov::hint::model_priority(ov::hint::Priority::LOW)},
                                                           {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetConfigTest,
                         OVClassCompiledModelGetConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelSetIncorrectConfigTest,
                         OVClassCompiledModelSetIncorrectConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

}  // namespace
