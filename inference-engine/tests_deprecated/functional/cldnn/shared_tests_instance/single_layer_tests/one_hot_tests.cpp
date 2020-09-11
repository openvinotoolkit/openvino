// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot_tests.hpp"

one_hot_test_params one_hot_only_4d_test_cases[] = {
        one_hot_test_params("GPU", case_2d_0),
        one_hot_test_params("GPU", case_2d_1),
        one_hot_test_params("GPU", case_2d_2),
        one_hot_test_params("GPU", case_3d_0),
        one_hot_test_params("GPU", case_3d_1),
        one_hot_test_params("GPU", case_3d_2),
        one_hot_test_params("GPU", case_4d_0),
        one_hot_test_params("GPU", case_4d_1),
        one_hot_test_params("GPU", case_4d_2),
        one_hot_test_params("GPU", case_4d_3),
        one_hot_test_params("GPU", case_5d_0),
        one_hot_test_params("GPU", case_5d_1),
        one_hot_test_params("GPU", case_5d_2),
        one_hot_test_params("GPU", case_5d_3),
        one_hot_test_params("GPU", case_5d_4)
};

INSTANTIATE_TEST_CASE_P(nightly_TestsOneHot, OneHotOnlyTestShared, ::testing::ValuesIn(one_hot_only_4d_test_cases));
