// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/remote.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_plugin_config.hpp"

using namespace ov::test;

namespace {
const std::vector<ov::AnyMap> configs;


std::vector<std::pair<ov::AnyMap, ov::AnyMap>> generate_remote_params() {
        return {};
}

auto AutoBatchConfigs = []() {
    return std::vector<ov::AnyMap>{
        // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_GPU) + "(4)"},
         // no timeout to avoid increasing the test time
         {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "0 "}}};
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_BehaviorTests, OVRemoteTest,
                        ::testing::Combine(
                                ::testing::Values(ngraph::element::f32),
                                ::testing::Values(::ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs),
                                ::testing::ValuesIn(generate_remote_params())),
                        OVRemoteTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_AutoBatch_BehaviorTests, OVRemoteTest,
                         ::testing::Combine(
                                 ::testing::Values(ngraph::element::f32),
                                 ::testing::Values(::ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs()),
                                 ::testing::ValuesIn(generate_remote_params())),
                         OVRemoteTest::getTestCaseName);
} // namespace
