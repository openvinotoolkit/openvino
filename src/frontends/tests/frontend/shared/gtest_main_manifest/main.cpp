// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "gtest/gtest.h"
#include "utils.hpp"
#include "common_test_utils/file_utils.hpp"

#include "ngraph/file_util.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
static const std::string s_manifest{
#ifdef MANIFEST
    ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(), MANIFEST)
#endif
};
OPENVINO_SUPPRESS_DEPRECATED_END

int main(int argc, char** argv) {
    printf("Running main() from %s:%d\n", __FILE__, __LINE__);
    return FrontEndTestUtils::run_tests(argc, argv, s_manifest);
}
