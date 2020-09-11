// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_tests.hpp"


activation_test_params act_test_cases[] = {
        activation_test_params("CPU", case_1, "relu"),
        activation_test_params("CPU", case_1, "exp"),
        activation_test_params("CPU", case_1, "not"),
};

INSTANTIATE_TEST_CASE_P(
        smoke_CPU_TestsActivationFunctions, ActivationTest, ::testing::ValuesIn(act_test_cases), getTestCaseName);
