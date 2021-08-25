// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/executable_network/exec_graph_info.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        ExecutableNetworkBaseTest::getTestCaseName);

}  // namespace