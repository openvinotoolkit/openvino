// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/life_time.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<std::vector<int>> orders = {
    // 0 - plugin
    // 1 - executable_network
    // 2 - infer_request
    {0, 1, 2},
    {0, 2, 1},
    {1, 0, 2},
    {1, 2, 0},
    {2, 0, 1},
    {2, 1, 0}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         HoldersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(orders)),
                         HoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         HoldersTestImportNetwork,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE, "HETERO:TEMPLATE"),
                                            ::testing::ValuesIn(orders)),
                         HoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         HoldersTestOnImportedNetwork,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE, "HETERO:TEMPLATE"),
                         HoldersTestOnImportedNetwork::getTestCaseName);

}  // namespace
