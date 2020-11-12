// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

namespace FuncTestUtils {
namespace SkipTestsConfig {

bool disable_tests_skipping = false;

bool currentTestIsDisabled() {
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
