// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_CPU_TestEltwise, EltwiseOnlyTest,
        ::testing::Values(
                eltwise_test_params{"CPU",
                                    {13, 13, 1}, eltwise_test_params::Sum, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Max, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Prod, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Sub, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Min, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Div, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Squared_diff, 2},
                eltwise_test_params{"CPU",
                                    {13, 13, 1}, eltwise_test_params::Equal, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Not_equal, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Less, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Less_equal, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Greater, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Greater_equal, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Logical_AND, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Logical_OR, 2},
                eltwise_test_params{"CPU",
                                    {23, 23, 1}, eltwise_test_params::Logical_XOR, 2}
        ));
