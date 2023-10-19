// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
#ifdef OPENVINO_STATIC_LIBRARY
        // Disable tests for static libraries
        ".*FrontendLibCloseTest.*"
#endif
    };
}
