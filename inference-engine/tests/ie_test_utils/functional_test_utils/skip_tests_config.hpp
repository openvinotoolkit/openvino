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

inline bool currentTestIsDisabled() {
    const auto fullName = ::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()
                          + std::string(".") + ::testing::UnitTest::GetInstance()->current_test_info()->name();
    for (const auto &pattern : disabledTestPatterns()) {
        std::regex re(pattern);
        if (std::regex_match(fullName, re))
            return true;
    }
    return false;
}

}  // namespace SkipTestsConfig
}  // namespace FuncTestUtils

#define SKIP_IF_CURRENT_TEST_IS_DISABLED()                              \
{                                                                       \
    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {      \
        SKIP() << "Disabled test due to configuration" << std::endl;    \
    }                                                                   \
}
