// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "functional_test_utils/layer_test_utils.hpp"

bool ignore_skips;
int main(int argc, char* argv[]) {
    ignore_skips = false;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--disable_test_skips") {
            ignore_skips = true;
        }
    }
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new LayerTestsUtils::TestEnvironment);
    auto retcode = RUN_ALL_TESTS();

    return retcode;
}