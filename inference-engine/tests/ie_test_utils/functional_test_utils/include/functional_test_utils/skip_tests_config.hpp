// Copyright (C) 2018-2021 Intel Corporation
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

extern bool disable_tests_skipping;

bool currentTestIsDisabled();
std::vector<std::string> readSkipTestConfigFiles(const std::vector<std::string>& filePaths);

}  // namespace SkipTestsConfig
}  // namespace FuncTestUtils

#define SKIP_IF_CURRENT_TEST_IS_DISABLED()                              \
{                                                                       \
    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {      \
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;    \
    }                                                                   \
}
