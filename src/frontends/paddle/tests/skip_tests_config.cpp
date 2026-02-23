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
        std::regex(".*testUnloadLibBeforeDeletingDependentObject.*"),
        // CVS-130605, CVS-170348
        std::regex(".*paddle_yolo_box_uneven_wh_yolo_box_uneven_wh_pdmodel.*"),
        std::regex(".*paddle_loop_dyn_loop_dyn_pdmodel.*"),
        std::regex(".*paddle_scatter_test_1_scatter_test_1_pdmodel.*"),
        std::regex(".*paddle_top_k_.*"),
        std::regex(".*generate_proposals.*"),
    };

    return patterns;
}
