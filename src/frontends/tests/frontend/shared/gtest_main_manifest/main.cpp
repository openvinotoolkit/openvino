// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/util/file_util.hpp"
#include "utils.hpp"

static const std::string s_manifest{
#ifdef MANIFEST
    ov::util::path_join({ov::test::utils::getExecutableDirectory(), MANIFEST})
#endif
};

int main(int argc, char** argv) {
    printf("Running main() from %s:%d\n", __FILE__, __LINE__);
    return FrontEndTestUtils::run_tests(argc, argv, s_manifest);
}
