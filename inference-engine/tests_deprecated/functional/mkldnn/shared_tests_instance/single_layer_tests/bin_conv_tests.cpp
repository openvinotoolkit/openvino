// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bin_conv_tests.hpp"

bin_conv_test_params bin_conv_only_test_cases[] = {
        bin_conv_test_params("CPU", case_1),
        bin_conv_test_params("CPU", case_2),
        bin_conv_test_params("CPU", case_3),
        bin_conv_test_params("CPU", case_4),
        bin_conv_test_params("CPU", case_5),
        bin_conv_test_params("CPU", case_6),
        bin_conv_test_params("CPU", case_7),
        bin_conv_test_params("CPU", case_8),
        bin_conv_test_params("CPU", case_9),
        bin_conv_test_params("CPU", case_10),
        bin_conv_test_params("CPU", case_11),
        bin_conv_test_params("CPU", case_12),
        bin_conv_test_params("CPU", case_13),
        bin_conv_test_params("CPU", case_14),
        bin_conv_test_params("CPU", case_15),
        bin_conv_test_params("CPU", case_16)
};

INSTANTIATE_TEST_CASE_P(
        smoke_CPU_TestBinaryConvolution, BinaryConvolutionOnlyTest, ::testing::ValuesIn(bin_conv_only_test_cases), getTestCaseName);
