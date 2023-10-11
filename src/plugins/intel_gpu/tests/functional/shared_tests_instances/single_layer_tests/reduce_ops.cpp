// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reduce_ops.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8,
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

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
        ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::Prod,
        ngraph::helpers::ReductionType::L1,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<ngraph::helpers::ReductionType> reductionLogicalTypes = {
        ngraph::helpers::ReductionType::LogicalOr,
        ngraph::helpers::ReductionType::LogicalAnd
};

INSTANTIATE_TEST_SUITE_P(smoke_ReduceOneAxis,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0}),
                                          testing::ValuesIn(opTypes),
                                          testing::Values(true, false),
                                          testing::ValuesIn(reductionTypes),
                                          testing::Values(InferenceEngine::Precision::FP32),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::ValuesIn(inputShapesOneAxis),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceLogicalOneAxis,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0}),
                                          testing::ValuesIn(opTypes),
                                          testing::Values(true, false),
                                          testing::ValuesIn(reductionLogicalTypes),
                                          testing::Values(InferenceEngine::Precision::BOOL),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::ValuesIn(inputShapes),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Precisions,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{1, 3}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::Values(ngraph::helpers::ReductionType::Sum),
                                          testing::Values(InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::I32),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Ranks,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{2, 3, 4},
                                                          std::vector<int>{-1, -2}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::Values(ngraph::helpers::ReductionType::Sum),
                                          testing::Values(InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::I32),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::Values(std::vector<size_t>{2, 3, 4, 5, 4, 3, 2, 3},
                                                          std::vector<size_t>{2, 3, 4, 5, 4, 3, 2}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_InputShapes,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::Values(ngraph::helpers::ReductionType::Mean),
                                          testing::Values(InferenceEngine::Precision::FP32),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
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
                                          testing::Values(ngraph::helpers::ReductionType::Mean),
                                          testing::Values(InferenceEngine::Precision::FP32),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::ValuesIn(inputShapes),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_ReductionTypes,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0, 1, 3}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::ValuesIn(reductionTypes),
                                          testing::Values(InferenceEngine::Precision::FP32),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::Values(std::vector<size_t>{2, 9, 2, 9}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReduceLogical_ReductionTypes,
                         ReduceOpsLayerTest,
                         testing::Combine(testing::Values(std::vector<int>{0, 1, 3}),
                                          testing::Values(opTypes[1]),
                                          testing::ValuesIn(keepDims),
                                          testing::ValuesIn(reductionLogicalTypes),
                                          testing::Values(InferenceEngine::Precision::BOOL),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::Values(std::vector<size_t>{2, 9, 2, 9}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce,
                         ReduceOpsLayerWithSpecificInputTest,
                         testing::Combine(testing::ValuesIn(decltype(axes){{0}, {1}}),
                                          testing::Values(opTypes[1]),
                                          testing::Values(true),
                                          testing::Values(ngraph::helpers::ReductionType::Sum),
                                          testing::Values(InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::I32),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::Values(std::vector<size_t>{2, 10}),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ReduceOpsLayerWithSpecificInputTest::getTestCaseName);

}  // namespace
