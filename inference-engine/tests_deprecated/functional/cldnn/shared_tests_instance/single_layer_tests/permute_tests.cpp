// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_tests.hpp"

permute_test_params permute_only_test_cases[] = {
        permute_test_params("GPU", case_1),
        permute_test_params("GPU", case_2),
        permute_test_params("GPU", case_3),
        permute_test_params("GPU", case_4),
        permute_test_params("GPU", case_5),
        permute_test_params("GPU", case_6),
        permute_test_params("GPU", case_7),
        permute_test_params("GPU", case_8),
        permute_test_params("GPU", case_9),
        permute_test_params("GPU", case_10),
        permute_test_params("GPU", case_11),
        permute_test_params("GPU", case_12),
        permute_test_params("GPU", case_13),
        permute_test_params("GPU", case_14),
        permute_test_params("GPU", case_15),
        permute_test_params("GPU", case_16)
};


INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestPermute, PermuteOnlyTests, ::testing::ValuesIn(permute_only_test_cases));

