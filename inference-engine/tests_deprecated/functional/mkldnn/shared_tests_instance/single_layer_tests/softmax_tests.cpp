// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_tests.hpp"

softmax_test_params softmax_only_test_cases[] = {
        softmax_test_params("CPU", case_1),
        softmax_test_params("CPU", case_8),
        softmax_test_params("CPU", case_8_nc, "2D"),
};

INSTANTIATE_TEST_CASE_P(
        smoke_CPU_TestsSoftmax, SoftmaxOnlyTest, ::testing::ValuesIn(softmax_only_test_cases)/*, getTestCaseName*/);
