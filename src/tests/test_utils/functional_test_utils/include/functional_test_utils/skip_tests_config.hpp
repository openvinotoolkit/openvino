// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <regex>
#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns();

bool is_model_cache_enabled();

const std::vector<std::regex>& model_cache_disabled_test_patterns();

namespace ov {
namespace test {
namespace utils {

extern bool disable_tests_skipping;

bool current_test_is_disabled();

bool is_model_cache_for_current_test_enabled();

}  // namespace utils
}  // namespace test
}  // namespace ov

#define SKIP_IF_CURRENT_TEST_IS_DISABLED()                                     \
    {                                                                          \
        if (ov::test::utils::current_test_is_disabled()) {                     \
            GTEST_SKIP() << "Disabled test due to configuration" << std::endl; \
        }                                                                      \
    }
