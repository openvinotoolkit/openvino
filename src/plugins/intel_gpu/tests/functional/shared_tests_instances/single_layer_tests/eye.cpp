/// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/eye.hpp"

namespace {
using ov::test::EyeLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32,
    ov::element::i8,
    ov::element::u8,
    ov::element::i64};

const std::vector<std::vector<int>> eyePars = {
    // rows, cols, diag_shift
    {3, 3, 0},
    {3, 4, 1},
    {4, 3, 1},
    {3, 4, 0},
    {4, 3, 0},
    {3, 4, -1},
    {4, 3, -1},
    {3, 4, 10},
    {4, 4, -2},
};

// dummy parameter to prevent empty set of test cases
const std::vector<std::vector<int>> emptyBatchShape = {{0}};
const std::vector<std::vector<int>> batchShapes1D = {{3}, {2}, {1}, {0}};
const std::vector<std::vector<int>> batchShapes2D = {{3, 2}, {2, 1}, {0, 0}};
const std::vector<std::vector<int>> batchShapes3D = {{3, 2, 1}, {1, 1, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_Eye2D_WithNonScalar_Test,
                         EyeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<ov::Shape>>{{{1}, {1}, {1}}}),
                                            ::testing::ValuesIn(emptyBatchShape),
                                            ::testing::ValuesIn(eyePars),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_1DBatch_Test,
                         EyeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<ov::Shape>>{
                                                {{1}, {1}, {1}, {1}}}),
                                            ::testing::ValuesIn(batchShapes1D),
                                            ::testing::ValuesIn(eyePars),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_2DBatch_Test,
                         EyeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<ov::Shape>>{
                                                {{1}, {1}, {1}, {2}}}),
                                            ::testing::ValuesIn(batchShapes2D),
                                            ::testing::ValuesIn(eyePars),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_3DBatch_Test,
                         EyeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<ov::Shape>>{
                                                {{1}, {1}, {1}, {3}}}),
                                            ::testing::ValuesIn(batchShapes3D),
                                            ::testing::ValuesIn(eyePars),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         EyeLayerTest::getTestCaseName);

}  // namespace
