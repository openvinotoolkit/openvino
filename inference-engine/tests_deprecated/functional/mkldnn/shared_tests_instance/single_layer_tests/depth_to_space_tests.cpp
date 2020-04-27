// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_tests.hpp"

//TEST_P(DepthToSpaceTests, TestsDepthToSpace) {}

//INSTANTIATE_TEST_CASE_P(
//        TestsDepthToSpace, DepthToSpaceTests,
//        ::testing::Values(
//        depth_to_space_test_params{ "CPU", "FP32", { 1, 4, 1, 1 }, input0, 2, { 1, 1, 2, 2 }, ref_input0_bs2 },
//        depth_to_space_test_params{ "CPU", "FP32", { 1, 4, 2, 1 }, input1, 2, { 1, 1, 4, 2 }, ref_input1_bs2 },
//        depth_to_space_test_params{ "CPU", "FP32", { 1, 4, 2, 2 }, input2, 2, { 1, 1, 4, 4 }, ref_input2_bs2 },
//        depth_to_space_test_params{ "CPU", "FP32", { 1, 4, 3, 2 }, input3, 2, { 1, 1, 6, 4 }, ref_input3_bs2 },
//        depth_to_space_test_params{ "CPU", "FP32", { 1, 9, 3, 3 }, input4, 3, { 1, 1, 9, 9 }, ref_input4_bs3 },
//        depth_to_space_test_params{ "CPU", "FP32", { 1, 18, 3, 3 }, input5, 3, { 1, 2, 9, 9 }, ref_input5_bs3 }
//));
