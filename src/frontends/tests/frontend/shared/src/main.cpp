// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "utils.hpp"

using namespace std;

namespace FrontEndTestUtils {
int run_tests(int argc, char** argv, const std::string& manifest) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!manifest.empty()) {
        ::testing::GTEST_FLAG(filter) += FrontEndTestUtils::get_disabled_tests(manifest);
    }
    return RUN_ALL_TESTS();
}
}  // namespace FrontEndTestUtils
