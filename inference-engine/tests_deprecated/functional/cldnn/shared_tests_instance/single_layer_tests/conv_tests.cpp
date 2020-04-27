// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_tests.hpp"

conv_test_params conv_only_test_cases[] = {
        conv_test_params("GPU", case_1),
        conv_test_params("GPU", case_2),
        conv_test_params("GPU", case_3),
        conv_test_params("GPU", case_4),
        conv_test_params("GPU", case_5),
        conv_test_params("GPU", case_6),
        conv_test_params("GPU", case_7),
        conv_test_params("GPU", case_8),
        conv_test_params("GPU", case_9),
        conv_test_params("GPU", case_10),
        conv_test_params("GPU", case_11),
        conv_test_params("GPU", case_12),
        conv_test_params("GPU", case_13),
        conv_test_params("GPU", case_14)
};

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestConvolution, ConvolutionOnlyTest, ::testing::ValuesIn(conv_only_test_cases), getTestCaseName);

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestConvolutionBlobsAsInputs, ConvolutionBlobsAsInputsTest, ::testing::ValuesIn(conv_only_test_cases), getTestCaseName);

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestConvolutionSameUpper, ConvolutionReshapeTest,
        ::testing::Values(conv_test_params("GPU", case_si_1)),
        getTestCaseName);
