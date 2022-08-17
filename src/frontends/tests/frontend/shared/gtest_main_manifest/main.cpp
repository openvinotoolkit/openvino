// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "gtest/gtest.h"
#include "utils.hpp"

static const std::string manifest{
#ifdef MANIFEST
    MANIFEST
#endif
};

int main(int argc, char** argv) {
    if (!manifest.empty()) {
        testing::GTEST_FLAG(filter) += FrontEndTestUtils::get_disabled_tests(manifest);
    }
    return FrontEndTestUtils::run_tests(argc, argv);
}
