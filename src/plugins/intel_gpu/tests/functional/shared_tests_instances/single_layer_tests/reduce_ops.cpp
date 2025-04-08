// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reduce_ops.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ReduceOpsLayerTest;
using ov::test::ReduceOpsLayerWithSpecificInputTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::i32,
        ov::element::u8,
        ov::element::i8,
};

const std::vector<bool> keepDims = {
        true,
        false,
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 20, 30, 40},
        std::vector<size_t>{3, 5, 7, 9},
};

const std::vector<std::vector<size_t>> inputShapesOneAxis = {
        std::vector<size_t>{2, 3, 4, 5, 4, 3, 2, 3},
        std::vector<size_t>{10, 20, 30, 40},
        std::vector<size_t>{3, 5, 7, 9},
        std::vector<size_t>{10},
};

const std::vector<std::vector<int>> axes = {
        {0},
        {1},
        {2},
        {3},
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3},
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3},
        {0, 1, 2, 3},
        {1, -1}
};

std::vector<ov::test::utils::OpType> opTypes = {
        ov::test::utils::OpType::SCALAR,
        ov::test::utils::OpType::VECTOR,
};

const std::vector<ov::test::utils::ReductionType> reductionTypes = {
        ov::test::utils::ReductionType::Mean,
        ov::test::utils::ReductionType::Min,
        ov::test::utils::ReductionType::Max,
        ov::test::utils::ReductionType::Sum,
        ov::test::utils::ReductionType::Prod,
        ov::test::utils::ReductionType::L1,
        ov::test::utils::ReductionType::L2,
};

const std::vector<ov::test::utils::ReductionType> reductionLogicalTypes = {
        ov::test::utils::ReductionType::LogicalOr,
        ov::test::utils::ReductionType::LogicalAnd
};

INSTANTIATE_TEST_SUITE_P(smoke_ReduceOneAxis,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0}),
                                          testing::ValuesIn(opTypes),
                                          testing::Values(true, false),
                                          testing::ValuesIn(reductionTypes),
                                          testing::Values(ov::element::f32),
                                          testing::ValuesIn(inputShapesOneAxis),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceLogicalOneAxis,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0}),
                                          testing::ValuesIn(opTypes),
                                          testing::Values(true, false),
                                          testing::ValuesIn(reductionLogicalTypes),
                                          testing::Values(ov::element::boolean),
                                          testing::ValuesIn(inputShapes),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Precisions,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{1, 3}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::Values(ov::test::utils::ReductionType::Sum),
                                          testing::Values(ov::element::f32,
                                                          ov::element::i32),
                                          testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Ranks,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{2, 3, 4},
                                                          std::vector<int>{-1, -2}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::Values(ov::test::utils::ReductionType::Sum),
                                          testing::Values(ov::element::f32,
                                                          ov::element::i32),
                                          testing::Values(std::vector<size_t>{2, 3, 4, 5, 4, 3, 2, 3},
                                                          std::vector<size_t>{2, 3, 4, 5, 4, 3, 2}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_InputShapes,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::Values(ov::test::utils::ReductionType::Mean),
                                          testing::Values(ov::element::f32),
                                          testing::Values(std::vector<size_t>{3},
                                                          std::vector<size_t>{3, 5},
                                                          std::vector<size_t>{2, 4, 6},
                                                          std::vector<size_t>{2, 4, 6, 8},
                                                          std::vector<size_t>{2, 2, 2, 2, 2},
                                                          std::vector<size_t>{2, 2, 2, 2, 2, 2}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Axes,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::ValuesIn(axes),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::Values(ov::test::utils::ReductionType::Mean),
                                          testing::Values(ov::element::f32),
                                          testing::ValuesIn(inputShapes),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_ReductionTypes,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0, 1, 3}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::ValuesIn(reductionTypes),
                                          testing::Values(ov::element::f32),
                                          testing::Values(std::vector<size_t>{2, 9, 2, 9}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceLogical_ReductionTypes,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0, 1, 3}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::ValuesIn(reductionLogicalTypes),
                                          testing::Values(ov::element::boolean),
                                          testing::Values(std::vector<size_t>{2, 9, 2, 9}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce,
                         ReduceOpsLayerWithSpecificInputTest,
                         testing::Combine(testing::ValuesIn(decltype(axes){{0}, {1}}),
                                          testing::Values(opTypes[1]),
                                          testing::Values(true),
                                          testing::Values(ov::test::utils::ReductionType::Sum),
                                          testing::Values(ov::element::f32,
                                                          ov::element::i32),
                                          testing::Values(std::vector<size_t>{2, 10}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerWithSpecificInputTest::getTestCaseName);

}  // namespace
