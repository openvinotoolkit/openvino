// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: FIX BUG 31661
        ".*Behavior.*CallbackThrowException.*"
        ".*Behavior.*Callback.*",
        // TODO: Issue 32541
        ".*MultiplyLayerTest.*secondaryInputType=PARAMETER.*",
        // TODO: Issue 32544
        ".*non_flat.*(Add|Subtract)LayerTest.*",
        // TODO: Issue 32542
        ".*(Add|Subtract)LayerTest.*Type=SCALAR.*",
        // TODO: Issue 32521
        ".*SubtractLayerTest.*netPRC=FP16.*"
    };
}
