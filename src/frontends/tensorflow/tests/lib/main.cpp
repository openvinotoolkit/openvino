// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "utils.hpp"

int main(int argc, char** argv) {
    testing::GTEST_FLAG(filter) += FrontEndTestUtils::get_disabled_tests(std::string(MANIFEST));
    return FrontEndTestUtils::run_tests(argc, argv);
}
