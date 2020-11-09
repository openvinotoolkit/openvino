// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <regex>
#include <gtest/gtest.h>

std::vector<std::string> disabledTestPatterns();

namespace FuncTestUtils {
namespace SkipTestsConfig {

bool disable_tests_skipping = false;

inline bool currentTestIsDisabled() {
    bool skip_test = false;
    const auto fullName = ::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()
                          + std::string(".") + ::testing::UnitTest::GetInstance()->current_test_info()->name();
    for (const auto &pattern : disabledTestPatterns()) {
        std::regex re(pattern);
        if (std::regex_match(fullName, re))
            skip_test = true;
    }
    return skip_test && !disable_tests_skipping;
}

}  // namespace SkipTestsConfig
}  // namespace FuncTestUtils

#define SKIP_IF_CURRENT_TEST_IS_DISABLED()                              \
{                                                                       \
    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {      \
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;    \
    }                                                                   \
}
