// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <regex>
#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns();

namespace ov {
namespace test {
namespace utils {

extern bool disable_tests_skipping;

bool current_test_is_disabled();

}  // namespace utils
}  // namespace test
}  // namespace ov

// TODO: Remove after migration of internal components
namespace FuncTestUtils {
namespace SkipTestsConfig {

inline bool currentTestIsDisabled() {
    return ov::test::utils::current_test_is_disabled();
}

}  // namespace SkipTestsConfig
}  // namespace FuncTestUtils

#define SKIP_IF_CURRENT_TEST_IS_DISABLED()                                     \
    {                                                                          \
        if (ov::test::utils::current_test_is_disabled()) {                     \
            GTEST_SKIP() << "Disabled test due to configuration" << std::endl; \
        }                                                                      \
    }
