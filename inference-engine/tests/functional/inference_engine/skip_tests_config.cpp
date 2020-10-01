// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: Issue: 33375
        // Disabled due to rare sporadic failures.
        ".*TransformationTests.*ConstFoldingPriorBoxClustered.*",
        // TODO: Issue: 32568, enable after supporting constants outputs in plugins
        ".*TransformationTests.*ConstFoldingPriorBox.*",
    };
}
