// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

const std::vector<std::regex>& disabled_test_patterns() {
    const static std::vector<std::regex> patterns{
#ifdef OPENVINO_STATIC_LIBRARY
        // Disable tests for static libraries
        std::regex(".*FrontendLibCloseTest.*"),
#endif
        // CVS-123201
        std::regex(".*testUnloadLibBeforeDeletingDependentObject.*"),
    };

    return patterns;
}
