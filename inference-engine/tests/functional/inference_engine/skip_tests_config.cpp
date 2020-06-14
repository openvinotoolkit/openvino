// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: FIX BUG 33375
        // Disabled due to rare sporadic failures.
        ".*TransformationTests\\.ConstFoldingPriorBoxClustered.*",
    };
}
