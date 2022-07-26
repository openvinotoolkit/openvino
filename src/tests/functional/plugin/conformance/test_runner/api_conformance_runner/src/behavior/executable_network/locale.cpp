// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/locale.hpp"
#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    INSTANTIATE_TEST_SUITE_P(ie_executable_network, CustomLocaleTest,
                            ::testing::Combine(
                                ::testing::Values("ru_RU.UTF-8"),
                                ::testing::ValuesIn(ov::test::conformance::return_all_possible_device_combination())),
                            CustomLocaleTest::getTestCaseName);
}  // namespace
