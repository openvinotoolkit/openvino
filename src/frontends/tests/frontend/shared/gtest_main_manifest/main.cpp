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
    return FrontEndTestUtils::run_tests(argc, argv, manifest);
}
