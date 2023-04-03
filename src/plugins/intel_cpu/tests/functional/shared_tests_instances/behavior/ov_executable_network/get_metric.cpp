// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <base/ov_behavior_test_utils.hpp>

#include "openvino/core/any.hpp"
#include "openvino/runtime/properties.hpp"

#include <gtest/gtest.h>

using namespace ov::test::behavior;
namespace {

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)),
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY::getTestCaseName);

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO:CPU"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));

} // namespace
