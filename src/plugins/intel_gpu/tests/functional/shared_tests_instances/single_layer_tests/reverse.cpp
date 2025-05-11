// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reverse.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ReverseLayerTest;

namespace {

const std::vector<ov::element::Type> model_types = {ov::element::f32,
                                                    ov::element::f16,
                                                    ov::element::i32,
                                                    ov::element::i64,
                                                    ov::element::i8,
                                                    ov::element::u8};

const std::vector<std::vector<size_t>> input_shapes_1D = {{10}};
const std::vector<std::vector<int>> indices_1D = {{0}};
const std::vector<std::string> modes = {"index", "mask"};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse1D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_1D),
                                            ::testing::ValuesIn(indices_1D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> input_shapes_2D = {{3, 4}, {1, 3}};
const std::vector<std::vector<int>> indices_2D = {{0}, {1}};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse2D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_2D),
                                            ::testing::ValuesIn(indices_2D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> input_shapes_3D = {{1, 3, 4}, {2, 5, 6}};
const std::vector<std::vector<int>> indices_3D = {{0}, {0, 1}, {0, 2}};
INSTANTIATE_TEST_SUITE_P(smoke_Reverse3D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_3D),
                                            ::testing::ValuesIn(indices_3D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> input_shapes_4D = {{1, 2, 3, 4}, {1, 2, 5, 6}};
const std::vector<std::vector<int>> indices_4D = {{1}, {1, 2}, {1, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse4D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_4D),
                                            ::testing::ValuesIn(indices_4D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> input_shapes_5D = {{1, 1, 4, 3, 3}};
const std::vector<std::vector<int>> indices_5D = {{2}, {2, 3}, {2, 4}};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse5D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_5D),
                                            ::testing::ValuesIn(indices_5D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> input_shapes_6D = {{1, 1, 4, 3, 3, 3}};
const std::vector<std::vector<int>> indices_6D = {{2}, {1, 3}, {3, 5}, {1, 4, 5}};

INSTANTIATE_TEST_SUITE_P(smoke_Reverse6D,
                         ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_6D),
                                            ::testing::ValuesIn(indices_6D),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReverseLayerTest::getTestCaseName);

}  // namespace
