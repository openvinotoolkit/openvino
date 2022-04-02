// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: task 32568, enable after supporting constants outputs in plugins
        ".*TransformationTests\\.ConstFoldingPriorBox.*",
        // azure is failing after #6199
        ".*/NmsLayerTest.*"
    };
}
