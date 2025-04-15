// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: task 32568, enable after supporting constants outputs in plugins
        ".*TransformationTests\\.ConstFoldingPriorBox.*",
    };
}
