// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/version.hpp"
#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    INSTANTIATE_TEST_SUITE_P(ie_plugin, VersionTest,
                                    ::testing::ValuesIn(ov::test::conformance::return_all_possible_device_combination()),
                            VersionTest::getTestCaseName);
}  // namespace
