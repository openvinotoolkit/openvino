// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/infer_querystate.hpp"
#include <openvino/runtime/auto/properties.hpp>

using namespace ov::test::behavior;

namespace {
const auto configs = []() {
    return std::vector<ov::AnyMap>{};
};

const auto Multiconfigs = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};
};

const auto Autoconfigs = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestQueryStateTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::ValuesIn(configs())),
                         OVInferRequestQueryStateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVInferRequestQueryStateTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(Multiconfigs())),
                         OVInferRequestQueryStateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVInferRequestQueryStateTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(Autoconfigs())),
                         OVInferRequestQueryStateTest::getTestCaseName);
}  // namespace