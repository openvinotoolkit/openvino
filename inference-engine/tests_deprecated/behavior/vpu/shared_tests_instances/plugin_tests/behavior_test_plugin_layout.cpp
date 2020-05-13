// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layout.hpp"

layout_test_params power_test_cases[] = {
    layout_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", Layout::C, power_params({ { 3 } }, 2, 2, 2)),
    layout_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", Layout::NC, power_params({ { 1, 3 } }, 2, 2, 2)),
    layout_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", Layout::CHW, power_params({ { 3, 32, 16 } }, 2, 2, 2)),
    layout_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", Layout::NCHW, power_params({ { 1, 3, 16, 16 } }, 2, 2, 2)),
};
INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, LayoutTestCanLoadPower,
    ::testing::ValuesIn(power_test_cases), getTestName);

layout_test_params conv_neg_test_cases[] = {
    layout_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", Layout::C, power_params({ { 3 } }, 2, 2, 2)),
    layout_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", Layout::NC, power_params({ { 1, 3 } }, 2, 2, 2)),
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, LayoutTestCanNotLoadConv,
    ::testing::ValuesIn(conv_neg_test_cases), getTestName);

layout_test_params conv_test_cases[] = {
    layout_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", Layout::CHW, power_params({ { 3, 32, 16 } }, 2, 2, 2)),
    layout_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", Layout::NCHW, power_params({ { 1, 3, 16, 16 } }, 2, 2, 2)),
    };
    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, LayoutTestCanLoadConv,
    ::testing::ValuesIn(conv_test_cases), getTestName);
