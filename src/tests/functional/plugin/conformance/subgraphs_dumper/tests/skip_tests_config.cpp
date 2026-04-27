// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

const std::vector<std::regex>& disabled_test_patterns() {
    const static std::vector<std::regex> patterns{};
    return patterns;
}

bool is_model_cache_enabled() {
    return false;
}

std::vector<std::string> model_cache_disabled_test_patterns() {
    return std::vector<std::string>{};
}
