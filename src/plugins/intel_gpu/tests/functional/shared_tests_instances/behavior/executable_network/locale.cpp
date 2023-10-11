// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/locale.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    INSTANTIATE_TEST_SUITE_P(smoke_CustomLocaleTest, CustomLocaleTest,
                            ::testing::Combine(
                                ::testing::Values("ru_RU.UTF-8"),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                            CustomLocaleTest::getTestCaseName);
}  // namespace
