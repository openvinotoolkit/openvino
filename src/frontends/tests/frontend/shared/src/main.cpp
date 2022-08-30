// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

using namespace std;

namespace FrontEndTestUtils {
int run_tests(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();
    return rc;
}
}  // namespace FrontEndTestUtils