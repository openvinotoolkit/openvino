// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <regex>

#include <gtest/gtest.h>


std::vector<std::string> disabledTestPatterns();

namespace ov {
namespace test {
namespace utils {

extern bool disable_tests_skipping;

bool currentTestIsDisabled();

}  // namespace utils
}  // namespace test
}  // namespace ov

#define SKIP_IF_CURRENT_TEST_IS_DISABLED()                              \
{                                                                       \
    if (ov::test::utils::currentTestIsDisabled()) {      \
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;    \
    }                                                                   \
}
