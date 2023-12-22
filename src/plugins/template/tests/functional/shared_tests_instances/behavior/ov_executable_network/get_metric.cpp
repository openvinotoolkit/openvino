// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"
#include "behavior/compiled_model/properties_hetero.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

namespace {

//
// Executable Network GetMetric
//

std::vector<std::string> devices = {"TEMPLATE", "MULTI:TEMPLATE", "HETERO:TEMPLATE", "AUTO:TEMPLATE"};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::ValuesIn(devices));

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {{ov::hint::model_priority(ov::hint::Priority::HIGH)},
                                                           {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
                                                           {ov::hint::model_priority(ov::hint::Priority::LOW)},
                                                           {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO:TEMPLATE"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
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

////
//// Hetero Executable Network GetMetric
////
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroCompiledModelGetMetricTest,
                         OVClassHeteroCompiledModelGetMetricTest_TARGET_FALLBACK,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));
INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroCompiledModelGetMetricTest,
                         OVClassHeteroCompiledModelGetMetricTest_EXEC_DEVICES,
                         ::testing::Values("TEMPLATE.0"));
//////////////////////////////////////////////////////////////////////////////////////////

}  // namespace
