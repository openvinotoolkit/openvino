// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/exec_network_base.hpp"

using namespace ov::test::behavior;
namespace {
const std::vector<ov::AnyMap> configs = {
        {}
};
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVExecutableNetworkBaseTest,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                 ::testing::ValuesIn(configs)),
                         OVExecutableNetworkBaseTest::getTestCaseName);
}  // namespace//

