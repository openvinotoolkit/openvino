// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

namespace ov {
namespace test {
namespace behavior {
INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests,
                         OVHoldersTest,
                         ::testing::Values("HETERO:TEMPLATE"),
                         OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_VirtualPlugin_BehaviorTests,
                         OVHoldersTest,
                         ::testing::Values("HETERO:GPU"),
                         OVHoldersTest::getTestCaseName);
}  // namespace behavior
}  // namespace test
}  // namespace ov
