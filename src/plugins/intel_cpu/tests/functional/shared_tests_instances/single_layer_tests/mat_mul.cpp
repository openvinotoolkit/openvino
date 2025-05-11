// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/mat_mul.hpp"

namespace {
using ov::test::MatMulLayerTest;
using ov::test::utils::InputLayerType;

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::i32,
};

std::vector<std::vector<ov::Shape>> input_shapes_no_transpose_static {
    { {1, 4, 5, 6}, {1, 4, 6, 4} },
    { {4, 5, 6}, {6, 3} },
    { {9, 9, 9}, {9, 9} },
    { {1, 2, 3}, {1, 3, 10} },
    { {1, 2, 3}, {1, 1, 3, 2} },
    { {1, 3, 2, 4}, {2, 1, 4, 2} },
    { {2, 1, 2, 4}, {1, 3, 4, 2} },
    { {3, 2, 4}, {2, 1, 4, 2} },
    { {2, 1, 4, 2}, {3, 2, 4} },
    { {3}, {2, 2, 3, 1} },
    { {2, 2, 1, 3}, {3} },
    { {1, 5}, {5, 1} },
    { {1, 5}, {5} },
    { {5}, {5, 1} },
    { {5}, {5} },
};

std::vector<std::vector<ov::Shape>> input_shapes_first_transpose_static {
    { {2, 1, 2, 3}, {3, 2, 4} },
    { {5, 1}, {5, 1} },
};

std::vector<std::vector<ov::Shape>> input_shapes_second_transpose_static {
    { {1, 2, 3}, {1, 10, 3} },
    { {2, 1, 3, 2}, {3, 4, 2} },
    { {1, 5}, {10, 5} },
};

std::vector<std::vector<ov::Shape>> input_shapes_both_transpose_static {
    { {2, 1, 2, 3}, {3, 4, 2} },
    { {5}, {5}, },
};


std::vector<InputLayerType> secondary_input_types = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_NoTranspose, MatMulLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_no_transpose_static)),
                ::testing::Values(std::make_pair(false, false)),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(secondary_input_types),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        MatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_FirstTranspose, MatMulLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_first_transpose_static)),
                ::testing::Values(std::make_pair(true, false)),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(secondary_input_types),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        MatMulLayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_MatMul_SecondTranspose, MatMulLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_second_transpose_static)),
                ::testing::Values(std::make_pair(false, true)),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(secondary_input_types),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        MatMulLayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_MatMul_BothTranspose, MatMulLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_both_transpose_static)),
                ::testing::Values(std::make_pair(true, true)),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(secondary_input_types),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        MatMulLayerTest::getTestCaseName);

} // namespace

