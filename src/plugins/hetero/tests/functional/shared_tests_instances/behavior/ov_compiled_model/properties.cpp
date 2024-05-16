// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"

#include "behavior/compiled_model/properties_hetero.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> hetero_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::enable_profiling(true)},
    {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE), ov::device::id("0")},
};

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(hetero_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest,
                         ::testing::Values("HETERO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroCompiledModelGetMetricTest,
                         OVClassHeteroCompiledModelGetMetricTest_TARGET_FALLBACK,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));
INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroCompiledModelGetMetricTest,
                         OVClassHeteroCompiledModelGetMetricTest_EXEC_DEVICES,
                         ::testing::Values("TEMPLATE.0"));
}  // namespace
