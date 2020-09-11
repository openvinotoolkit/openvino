// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_tests.hpp"

activation_test_params test_cases[] = {
        activation_test_params("GPU", case_1, "relu"),
        activation_test_params("GPU", case_1, "exp"),
        activation_test_params("GPU", case_1, "not"),
        activation_test_params("GPU", case_1, "sin"),
        activation_test_params("GPU", case_1, "sinh"),
        activation_test_params("GPU", case_1, "cos"),
        activation_test_params("GPU", case_1, "cosh"),
};

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestsActivationFunctions, ActivationTest, ::testing::ValuesIn(test_cases), getTestCaseName);
