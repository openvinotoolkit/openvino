// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: Issue: 32521
        R"(.*(EltwiseLayerTest).*secondaryInputType=CONSTANT.*netPRC=FP16.*)",
        R"(.*(EltwiseLayerTest).*IS=.*1.1.1.1.*opType=SCALAR.*)"
    };
}