// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layout.hpp"

layout_test_params activ_test_cases[] = {
//    layout_test_params(CommonTestUtils::DEVICE_GNA, "FP32", Layout::C, power_params({ { 3 } }, 1, 2, 2)),
    layout_test_params(CommonTestUtils::DEVICE_GNA, "FP32", Layout::NC, power_params({ { 1, 3 } }, 1, 2, 2)),
    layout_test_params(CommonTestUtils::DEVICE_GNA, "FP32", Layout::CHW, power_params({ { 3, 32, 16 } }, 1, 2, 2)),
    layout_test_params(CommonTestUtils::DEVICE_GNA, "FP32", Layout::NCHW, power_params({ { 1, 3, 16, 16 } }, 2, 2, 2)),
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, LayoutTestCanLoadActiv,
    ::testing::ValuesIn(activ_test_cases), getTestName);
