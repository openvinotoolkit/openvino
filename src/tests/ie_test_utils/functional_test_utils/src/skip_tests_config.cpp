// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <fstream>

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace FuncTestUtils {
namespace SkipTestsConfig {

bool disable_tests_skipping = false;

bool currentTestIsDisabled() {
    if (disable_tests_skipping)
        return false;

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
