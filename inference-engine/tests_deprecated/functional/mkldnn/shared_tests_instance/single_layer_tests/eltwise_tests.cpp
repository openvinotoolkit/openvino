// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_CPU_TestEltwise, EltwiseOnlyTest,
        ::testing::Values(
                eltwise_test_params{"CPU",
                                    {13, 13, 1}, eltwise_test_params::Sum, 4},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Max, 3},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Prod, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Sub, 4},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Min, 3},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Div, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Squared_diff, 2},
                eltwise_test_params{"CPU",
                                    {13, 13, 1}, eltwise_test_params::Equal, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Not_equal, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Less, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Less_equal, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Greater, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Greater_equal, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Logical_AND, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Logical_OR, 5},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Logical_XOR, 5}
        ));
