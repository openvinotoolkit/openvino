// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layout.hpp"

layout_test_params power_test_cases[] = {
    layout_test_params("TEMPLATE", "FP16", Layout::NCHW, power_params({ { 1, 3, 16, 16 } }, 1, 2, 2)),
};

layout_test_params conv_test_cases_1[] = {
    layout_test_params("TEMPLATE", "FP16", Layout::NCHW, power_params({ { 1, 3, 16, 16 } }, 1, 2, 2)),
};

layout_test_params power_neg_test_cases[] = {
    // Graph Error Description: Error: Tensor size should not be 0.
    layout_test_params("TEMPLATE", "FP16", Layout::NC, power_params({ { 1, 3 } }, 1, 2, 2)),
    layout_test_params("TEMPLATE", "FP16", Layout::CHW, power_params({ { 3, 32, 16 } }, 1, 2, 2)),
};

layout_test_params conv_neg_test_cases[] = {
    // LoadNetwork hangs if Network has 1 dims format: CVS-8508
    layout_test_params("TEMPLATE", "FP16", Layout::C, power_params({ { 3 } }, 1, 2, 2)),
    layout_test_params("TEMPLATE", "FP16", Layout::NC, power_params({ { 1, 3 } }, 1, 2, 2)),
    layout_test_params("TEMPLATE", "FP16", Layout::CHW, power_params({ { 3, 32, 16 } }, 1, 2, 2)),
};

INSTANTIATE_TEST_CASE_P(BehaviorTest, LayoutTestCanLoadPower,
    ::testing::ValuesIn(power_test_cases), getTestName);
INSTANTIATE_TEST_CASE_P(BehaviorTest, LayoutTestCanLoadConv,
    ::testing::ValuesIn(conv_test_cases_1), getTestName);

    INSTANTIATE_TEST_CASE_P(BehaviorTest, LayoutTestCanNotLoadPower,
        ::testing::ValuesIn(power_neg_test_cases), getTestName);
INSTANTIATE_TEST_CASE_P(BehaviorTest, LayoutTestCanNotLoadConv,
    ::testing::ValuesIn(conv_neg_test_cases), getTestName);
