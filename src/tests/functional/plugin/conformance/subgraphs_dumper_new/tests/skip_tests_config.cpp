// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // todo: enable these tests with graph cache enabling
        R"(.*RepeatPatternExtractorTest.*extract_1.*)",
        R"(.*ModelUtilsTest.*generate_.*)",
        R"(.*GraphCacheFuncTest.*update_cache.*)",
        R"(.*FusedNamesExtractorTest.*)",
    };
    return retVector;
}
