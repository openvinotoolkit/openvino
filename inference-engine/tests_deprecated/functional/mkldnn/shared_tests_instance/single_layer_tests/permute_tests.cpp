// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_tests.hpp"

permute_test_params permute_only_test_cases[] = {
        permute_test_params("CPU", case_1),
        permute_test_params("CPU", case_2),
        permute_test_params("CPU", case_3),
        permute_test_params("CPU", case_4),
        permute_test_params("CPU", case_5),
        permute_test_params("CPU", case_6),
        permute_test_params("CPU", case_7),
        permute_test_params("CPU", case_8),
        permute_test_params("CPU", case_9),
        permute_test_params("CPU", case_10),
        permute_test_params("CPU", case_11),
        permute_test_params("CPU", case_12),
        permute_test_params("CPU", case_13),
        permute_test_params("CPU", case_14),
        permute_test_params("CPU", case_15),
        permute_test_params("CPU", case_16)
};


INSTANTIATE_TEST_CASE_P(
        smoke_CPU_TestPermute, PermuteOnlyTests, ::testing::ValuesIn(permute_only_test_cases));

