// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            // cldnn treats 1d constant as [1, f, 1, 1] tensor instead of [b, 1, 1, 1] which leads to fails of these tests
            R"(.*(EltwiseLayerTest).*IS=\(.*\..*\..*\..*\..*\).*secondaryInputType=CONSTANT.*opType=SCALAR.*)",
            R"(.*(EltwiseLayerTest).*IS=\(.*\).*secondaryInputType=CONSTANT.*)",
            // Issues - 34059
            ".*BehaviorTests\\.pluginDoesNotChangeOriginalNetwork.*"
    };
}