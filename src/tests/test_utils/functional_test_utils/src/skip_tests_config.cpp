// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <fstream>
#include <iostream>

#include "common_test_utils/file_utils.hpp"

namespace ov {
namespace test {
namespace utils {

bool disable_tests_skipping = false;

bool current_test_is_disabled() {
    if (disable_tests_skipping)
        return false;

    const auto fullName = ::testing::UnitTest::GetInstance()->current_test_info()->test_case_name() + std::string(".") +
                          ::testing::UnitTest::GetInstance()->current_test_info()->name();

    for (const auto& pattern : disabledTestPatterns()) {
        std::regex re(pattern);
        if (std::regex_match(fullName, re))
            return true;
    }

    return false;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
