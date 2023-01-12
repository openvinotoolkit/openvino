// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: for 22.2 (CVS-68949)
        R"(smoke_AutoBatching_CPU/AutoBatching_Test_DetectionOutput.*)",
    };
}
