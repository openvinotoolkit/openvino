// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    return {
#ifdef OPENVINO_STATIC_LIBRARY
        // Disable tests for static libraries
        ".*FrontendLibCloseTest.*",
#endif
        ".*testUnloadLibBeforeDeletingDependentObject.*",
        // CVS-130605, CVS-170348
        ".*paddle_yolo_box_uneven_wh_yolo_box_uneven_wh_pdmodel.*",
        ".*paddle_loop_dyn_loop_dyn_pdmodel.*",
        ".*paddle_scatter_test_1_scatter_test_1_pdmodel.*",
        ".*paddle_top_k_.*",
    };
}
