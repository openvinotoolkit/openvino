// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestsReduceSum, ReduceTestsShared,
        ::testing::Values(
        // Params: library_name, reduce_type, keep_dims, in_shape, input_tensor, axes_for_reduction, out_shape, reference
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ 0 },{ 1, 3, 4 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ -3 },{ 1, 3, 4 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ 2 },{ 2, 3, 1 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4, 1, 1 },{},{ 2 },{ 2, 3, 1, 1, 1 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ -1 },{ 2, 3, 1 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ 0, 2 },{ 1, 3, 1 },{ 68, 100, 132 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ 1, 2 },{ 2, 1, 1 },{ 78, 222 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ 2, 1 },{ 2, 1, 1 },{ 78, 222 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ 0, 1, 2 },{},{ 300 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSum", true,{ 2, 3, 4 },{},{ 0, -2, 2 },{},{ 300 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", true,{ 2, 3, 4 },{},{ 2, 2, 0, 2, 0 },{ 1, 3, 1 },{ 68, 100, 132 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ 0 },{ 3, 4 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ -3 },{ 3, 4 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ 2 },{ 2, 3 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ -1 },{ 2, 3 },{ 10, 26, 42, 58, 74, 90 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ 0, 2 },{ 3 },{ 68, 100, 132 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ 1, 2 },{ 2 },{ 78, 222 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ 2, 1 },{ 2 },{ 78, 222 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ 0, 1, 2 },{},{ 300 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ 0, -2, 2 },{},{ 300 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 2, 3, 4 },{},{ 2, 2, 0, 2, 0 },{ 3 },{ 68, 100, 132 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", true,{ 1, 2, 3, 4, 1 },{},{ 1 },{ 1, 1, 3, 4, 1 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } },
        reduce_test_params{ "GPU", "I32", "ReduceSum", false,{ 1, 2, 3, 4, 1 },{},{ 1 },{ 1, 3, 4, 1 },{ 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36 } }
));

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestsReduce, ReduceTestsShared,
        ::testing::Values(
        // Params: library_name, reduce_type, keep_dims, in_shape, input_tensor, axes_for_reduction, out_shape, reference
        reduce_test_params{ "GPU", "FP32", "ReduceAnd", true,{ 2, 2, 2 },{1, 0, 1, 1, 0, 1, 1, 0},{ 2 },{ 2, 2, 1 },{ 0, 1, 0, 0} },
        reduce_test_params{ "GPU", "FP32", "ReduceAnd", false, { 2, 2, 2 },{1, 0, 1, 1, 0, 1, 1, 0},{ 0, 1, 2 },{ },{ 0 } },
        reduce_test_params{ "GPU", "FP32", "ReduceL1", true,{ 10, 10, 2 },{},{ 2 },{ 10, 10, 1 },{ } },
        reduce_test_params{ "GPU", "FP32", "ReduceL1", true, { 3, 2, 2 },{},{ 2 },{ 3, 2, 1 },{ 3, 7, 11, 15, 19, 23 } },
        reduce_test_params{ "GPU", "FP32", "ReduceL1", false, { 3, 2, 2 },{},{ 2 },{ 3, 2 },{ 3, 7, 11, 15, 19, 23 } },
        reduce_test_params{ "GPU", "FP32", "ReduceL1", false, { 3, 2, 2 },{},{ 0, 1, 2 },{ },{ 78 } },
        reduce_test_params{ "GPU", "FP32", "ReduceL2", true,{ 10, 10, 2 },{},{ 2 },{ 10, 10, 1 },{} },
        reduce_test_params{ "GPU", "FP32", "ReduceL2", true,{ 3, 2, 2 },{},{ 2 },{ 3, 2, 1 },{ 2.23606798f, 5.f, 7.81024968f, 10.63014581f, 13.45362405f, 16.2788206f } },
        reduce_test_params{ "GPU", "FP32", "ReduceL2", false,{ 3, 2, 2 },{},{ 2 },{ 3, 2 },{ 2.23606798f, 5.f, 7.81024968f, 10.63014581f, 13.45362405f, 16.2788206f } },
        reduce_test_params{ "GPU", "FP32", "ReduceL2", false,{ 3, 2, 2 },{},{ 0, 1, 2 },{ },{ 25.49509757f } },
        reduce_test_params{ "GPU", "FP32", "ReduceLogSum", true,{ 10, 10, 2 },{},{ 2 },{ 10, 10, 1 },{} },
        reduce_test_params{ "GPU", "FP32", "ReduceLogSum", true,{ 3, 2, 2 },{ },{ 1 },{ 3, 1, 2 },{ } },
        reduce_test_params{ "GPU", "FP32", "ReduceLogSum", false,{ 3, 2, 2 },{ },{ 1 },{ 3, 2 },{ } },
        reduce_test_params{ "GPU", "FP32", "ReduceLogSum", false,{ 3, 2, 2 },{ },{ 0, 1, 2 },{},{ } },
        reduce_test_params{ "GPU", "FP32", "ReduceLogSumExp", true,{ 5, 5, 2 },{},{ 2 },{ 5, 5, 1 },{} },
        reduce_test_params{ "GPU", "FP32", "ReduceLogSumExp", true,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 20.f, 2.31326175f, 40.00004578f, 2.31326175f, 60.00671387f, 2.31326175f } },
        reduce_test_params{ "GPU", "FP32", "ReduceLogSumExp", false,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 20.f, 2.31326175f, 40.00004578f, 2.31326175f, 60.00671387f, 2.31326175f } },
        reduce_test_params{ "GPU", "FP32", "ReduceLogSumExp", false,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 0, 1, 2 },{},{ 60.00671387f } },
        reduce_test_params{ "GPU", "FP32", "ReduceMax", true,{ 10, 10, 2 },{},{ 2 },{ 10, 10, 1 },{} },
        reduce_test_params{ "GPU", "FP32", "ReduceMax", true,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 20, 2, 40, 2, 60, 2 } },
        reduce_test_params{ "GPU", "FP32", "ReduceMax", false,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 20, 2, 40, 2, 60, 2 } },
        reduce_test_params{ "GPU", "FP32", "ReduceMax", false,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 0, 1, 2 },{},{ 60 } },
        reduce_test_params{ "GPU", "FP32", "ReduceMean", true,{ 10, 10, 2 },{},{ 2 },{ 10, 10, 1 },{} },
        reduce_test_params{ "GPU", "FP32", "ReduceMean", true, { 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 12.5f, 1.5f, 35.f, 1.5f, 57.5f, 1.5f } },
        reduce_test_params{ "GPU", "FP32", "ReduceMean", false, { 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 12.5f, 1.5f, 35.f, 1.5f, 57.5f, 1.5f } },
        reduce_test_params{ "GPU", "FP32", "ReduceMean", false, { 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 0, 1, 2 },{ },{ 18.25f } },
        reduce_test_params{ "GPU", "FP32", "ReduceMin", true,{ 10, 10, 2 },{},{ 2 },{ 10, 10, 1 },{} },
        reduce_test_params{ "GPU", "FP32", "ReduceMin", true,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 1, 2 },{ 5, 1, 30, 1, 55, 1 } },
        reduce_test_params{ "GPU", "FP32", "ReduceMin", false,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 1 },{ 3, 2 },{ 5, 1, 30, 1, 55, 1 } },
        reduce_test_params{ "GPU", "FP32", "ReduceMin", false,{ 3, 2, 2 },{ 5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2 },{ 0, 1, 2 },{},{ 1 } },
        reduce_test_params{ "GPU", "FP32", "ReduceOr", true,{ 2, 2, 2 },{1, 0, 1, 1, 0, 0, 1, 0},{ 2 },{ 2, 2, 1 },{1, 1, 0, 1 } },
        reduce_test_params{ "GPU", "FP32", "ReduceOr", false, { 2, 2, 2 },{},{ 0, 1, 2 },{ },{ 1 } },
        reduce_test_params{ "GPU", "FP32", "ReduceProd", true,{ 10, 10, 2 },{},{ 2 },{ 10, 10, 1 },{} },
        reduce_test_params{ "GPU", "FP32", "ReduceProd", true,{ 3, 2, 2 },{},{ 1 },{ 3, 1, 2 },{ 3, 8, 35, 48, 99, 120 } },
        reduce_test_params{ "GPU", "FP32", "ReduceProd", false,{ 3, 2, 2 },{},{ 1 },{ 3, 2 },{ 3, 8, 35, 48, 99, 120 } },
        reduce_test_params{ "GPU", "FP32", "ReduceProd", false,{ 3, 2, 2 },{},{ 0, 1, 2 },{ },{ 4.790016e+08 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSumSquare", true,{ 10, 10, 2 },{},{ 2 },{ 10, 10, 1 },{} },
        reduce_test_params{ "GPU", "FP32", "ReduceSumSquare", true, { 3, 2, 2 },{},{ 1 },{ 3, 1, 2 },{ 10, 20, 74, 100, 202, 244 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSumSquare", false, { 3, 2, 2 },{},{ 1 },{ 3, 2 },{ 10, 20, 74, 100, 202, 244 } },
        reduce_test_params{ "GPU", "FP32", "ReduceSumSquare", false, { 3, 2, 2 },{},{ 0, 1, 2 },{ },{ 650 } }
));
