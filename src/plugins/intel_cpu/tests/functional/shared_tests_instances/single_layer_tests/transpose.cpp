// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::TransposeLayerTest;

namespace {
const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i64,
        ov::element::i32,
        ov::element::i16,
        ov::element::i8,
        ov::element::u8,
};

std::vector<std::vector<ov::Shape>> input_shape_static_2D = {{{2, 10}}, {{10, 2}}, {{10, 10}}};
std::vector<std::vector<size_t>> order_2D = {{0, 1}, {1, 0}};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose2D, TransposeLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(order_2D),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_2D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                TransposeLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> input_shape_static_4D = {{{2, 2, 2, 2}}, {{1, 10, 2, 3}}, {{2, 3, 4, 5}}};
std::vector<std::vector<size_t>> order_4D = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2},
    {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0},
    {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose4D, TransposeLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(order_4D),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_4D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                TransposeLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> input_shape_static_5D = {{{2, 2, 2, 2, 2}}, {{1, 10, 2, 3, 4}}, {{2, 3, 4, 5, 6}}};
std::vector<std::vector<size_t>> order_5D = {
    {0, 1, 2, 3, 4},
    {1, 0, 2, 3, 4},
    {4, 3, 2, 1, 0},
    {0, 2, 3, 4, 1},
    {1, 4, 2, 3, 0},
    {2, 4, 1, 0, 3},
    {3, 0, 2, 1, 4},
    {4, 1, 0, 3, 2},
    {0, 4, 1, 2, 3},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose5D, TransposeLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(order_5D),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_5D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                TransposeLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> input_shape_static_6D = {{{2, 2, 2, 2, 2, 2}}, {{1, 10, 2, 3, 4, 5}}, {{2, 3, 4, 5, 6, 7}}};
std::vector<std::vector<size_t>> order_6D = {
    {0, 1, 2, 3, 4, 5},
    {1, 0, 2, 3, 4, 5},
    {5, 4, 3, 2, 1, 0},
    {0, 2, 3, 4, 5, 1},
    {1, 5, 4, 2, 3, 0},
    {2, 5, 4, 1, 0, 3},
    {3, 0, 2, 1, 4, 5},
    {5, 1, 0, 4, 3, 2},
    {0, 5, 1, 2, 3, 4},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose6D, TransposeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(order_6D),
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_6D)),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        TransposeLayerTest::getTestCaseName);

}  // namespace
