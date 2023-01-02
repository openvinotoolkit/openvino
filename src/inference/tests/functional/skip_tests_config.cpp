// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
        R"(.*CachingTest.*TestNetworkModified.*ByCNNNetwork_testCache.*)",
        R"(.*CachingTest.*TestNetworkModified.*ByRemoteContext_testCache.*)",
        R"(.*CachingTest.*LoadHetero_MultiArchs.*ByCNNNetwork_testCache.*)",
        R"(.*CachingTest.*LoadHetero_MultiArchs.*ByModelName_testCache.*)",
    };
}
