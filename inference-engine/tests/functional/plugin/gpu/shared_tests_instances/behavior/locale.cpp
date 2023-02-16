// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/locale.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    INSTANTIATE_TEST_CASE_P(smoke_CustomLocaleTest, CustomLocaleTest,
                            ::testing::Combine(
                                ::testing::Values("ru_RU.UTF-8"),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                            CustomLocaleTest::getTestCaseName);
}  // namespace
