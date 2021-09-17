// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/stress_tests.hpp"

using namespace LayerTestsDefinitions;

const unsigned int g_BugAllocationLimit = 10000;

namespace {
    INSTANTIATE_TEST_SUITE_P(nightly_BehaviorTests, MultipleAllocations,
                            ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::Values(g_BugAllocationLimit)),
                            MultipleAllocations::getTestCaseName);
}  // namespace
