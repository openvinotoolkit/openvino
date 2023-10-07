// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/version.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         VersionTest,
                         ::testing::Values(ov::test::utils::DEVICE_MULTI),
                         VersionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         VersionTest,
                         ::testing::Values(ov::test::utils::DEVICE_AUTO),
                         VersionTest::getTestCaseName);
}  // namespace
