// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/einsum.hpp"

namespace {
using ov::test::EinsumLayerTest;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<ov::test::EinsumEquationWithInput> equationsWithInput = {
    { "ij->ji",           ov::test::static_shapes_to_test_representation({ {1, 2} }) }, // transpose 2d
    { "ijk->kij",         ov::test::static_shapes_to_test_representation({ {1, 2, 3} }) }, // transpose 3d
    { "ij->i",            ov::test::static_shapes_to_test_representation({ {2, 3} }) }, // reduce
    { "ab,cd->abcd",      ov::test::static_shapes_to_test_representation({ { 1, 2}, {3, 4} }) }, // no reduction
    { "ab,bc->ac",        ov::test::static_shapes_to_test_representation({ {2, 3}, {3, 2} }) }, // matrix multiplication
    { "ab,bcd,bc->ca",    ov::test::static_shapes_to_test_representation({ {2, 4}, {4, 3, 1}, {4, 3} }) },  // multiple multiplications
    { "kii->ki",          ov::test::static_shapes_to_test_representation({ {1, 3, 3} }) }, // diagonal
    { "abbac,bad->ad",    ov::test::static_shapes_to_test_representation({ {2, 3, 3, 2, 4}, {3, 2, 1} }) }, // diagonal and multiplication with repeated labels
    { "a...->...a",       ov::test::static_shapes_to_test_representation({ {2, 2, 3} }) }, // transpose with ellipsis
    { "a...->...",        ov::test::static_shapes_to_test_representation({ {2, 2, 3} }) }, // reduce with ellipsis
    { "ab...,...->ab...", ov::test::static_shapes_to_test_representation({ {2, 2, 3}, {1} }) }, // multiply by scalar
    { "a...j,j...->a...", ov::test::static_shapes_to_test_representation({ {1, 1, 4, 3}, {3, 4, 2, 1} }) } // complex multiplication
};

INSTANTIATE_TEST_SUITE_P(smoke_Einsum,
                         EinsumLayerTest,
                         ::testing::Combine(::testing::ValuesIn(model_types),
                                            ::testing::ValuesIn(equationsWithInput),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         EinsumLayerTest::getTestCaseName);

}  // namespace
