// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestsGather, GatherTFTests,
        ::testing::Values(
        gatherTF_test_params{ "GPU", "FP32", { 1, 4 }, in0,{ 2, 2 }, dict2D, 0, { 1, 4, 1, 2 }, ref_in0_a0_d22 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in0,{ 2, 2, 3 }, dict, 0, { 2, 2, 1, 2, 3 }, ref_in0_a0_d223 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in0,{ 2, 2, 3 }, dict,-3, { 2, 2, 1, 2, 3 }, ref_in0_a0_d223 },

        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in1,{ 3, 2, 2 }, dict, 0, { 2, 2, 1, 2, 2 }, ref_in1_a0_d322 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in1,{ 3, 2, 2 }, dict,-3, { 2, 2, 1, 2, 2 }, ref_in1_a0_d322 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in1,{ 2, 3, 2 }, dict, 1, { 2, 2, 2, 2, 1 }, ref_in1_a1_d232 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in1,{ 2, 3, 2 }, dict,-2, { 2, 2, 2, 2, 1 }, ref_in1_a1_d232 },

        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in1,{ 2, 2, 3 }, dict, 2, { 2, 2, 2, 2, 1 }, ref_in1_a2_d223 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in1,{ 2, 2, 3 }, dict,-1, { 2, 2, 2, 2, 1 }, ref_in1_a2_d223 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in0,{ 2, 3, 2 }, dict, 2, { 2, 3, 2, 2, 1 }, ref_in0_a2_d232 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in0,{ 2, 3, 2 }, dict,-1, { 2, 3, 2, 2, 1 }, ref_in0_a2_d232 },
        gatherTF_test_params{ "GPU", "FP32", { 2, 2 }, in0,{ 2, 3, 2 }, dict, 2, { 2, 3, 2, 2, 1 }, ref_in0_a2_d232 }
));
