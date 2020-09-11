// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestEltwise, EltwiseOnlyTest,
    ::testing::Values(
        eltwise_test_params{"GPU", {13, 13, 1}, eltwise_test_params::Sum, 5},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Max, 3},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Prod, 3},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Sub, 3},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Min, 7},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Div, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Squared_diff, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Equal, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Not_equal, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Less, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Less_equal, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Greater, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Greater_equal, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Logical_AND, 3},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Logical_OR, 4},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Logical_XOR, 4},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Floor_mod, 2},
        eltwise_test_params{"GPU", {23, 23, 1}, eltwise_test_params::Pow, 2}
        // TODO: Add tests for 1D/2D/3D blobs
));

/*** TBD ***/


