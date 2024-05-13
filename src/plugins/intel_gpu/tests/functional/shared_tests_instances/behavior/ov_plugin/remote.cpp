// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/remote.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {
const std::vector<ov::AnyMap> configs;


std::vector<std::pair<ov::AnyMap, ov::AnyMap>> generate_remote_params() {
        return {};
}

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_BehaviorTests, OVRemoteTest,
                        ::testing::Combine(
                                ::testing::Values(ov::element::f32),
                                ::testing::Values(::ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs),
                                ::testing::ValuesIn(generate_remote_params())),
                        OVRemoteTest::getTestCaseName);
} // namespace
