// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;
namespace {
INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests,
                         OVHoldersTest,
                         ::testing::Values("AUTO:TEMPLATE", "MULTI:TEMPLATE"),
                         OVHoldersTest::getTestCaseName);

const std::vector<std::string> device_names_and_priorities = {
    "MULTI:TEMPLATE",  // GPU via MULTI,
    "AUTO:TEMPLATE",   // GPU via AUTO,
};
INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests,
                         OVHoldersTestWithConfig,
                         ::testing::ValuesIn(device_names_and_priorities),
                         OVHoldersTestWithConfig::getTestCaseName);
}  // namespace
