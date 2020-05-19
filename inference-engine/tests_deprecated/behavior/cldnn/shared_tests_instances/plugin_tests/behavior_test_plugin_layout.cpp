// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layout.hpp"

layout_test_params power_test_cases[] = {
    layout_test_params("GPU", "FP32", Layout::NC, power_params({ { 1, 3 } }, 1, 2, 2)),
    layout_test_params("GPU", "FP32", Layout::NCHW, power_params({ { 1, 3, 16, 16 } }, 1, 2, 2)),
};

layout_test_params conv_test_cases[] = {
    layout_test_params("GPU", "FP32", Layout::NC, power_params({ { 1, 3 } }, 1, 2, 2)),
    layout_test_params("GPU", "FP32", Layout::NCHW, power_params({ { 1, 3, 16, 16 } }, 1, 2, 2)),
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, LayoutTestCanLoadPower,
    ::testing::ValuesIn(power_test_cases), getTestName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, LayoutTestCanLoadConv,
    ::testing::ValuesIn(conv_test_cases), getTestName);
