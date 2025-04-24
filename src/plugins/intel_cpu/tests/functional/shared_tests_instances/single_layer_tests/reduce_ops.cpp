// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/reduce_ops.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ReduceOpsLayerTest;
using ov::test::ReduceOpsLayerWithSpecificInputTest;

namespace {
const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i64,
        ov::element::i32,
        ov::element::u64
};

const std::vector<bool> keep_dims = {
        true,
        false,
};

const std::vector<std::vector<size_t>> input_shapes = {
        std::vector<size_t>{10, 20, 30, 40},
        std::vector<size_t>{3, 5, 7, 9},
};

const std::vector<std::vector<size_t>> input_shapes_0_dim = {
        std::vector<size_t>{2, 0, 4, 1},
        std::vector<size_t>{8, 0, 4, 0},
        std::vector<size_t>{2, 3, 4, 0},
};

const std::vector<std::vector<size_t>> input_shapes_one_axis = {
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

const std::vector<std::vector<int>> axes_0_dim = {
        {1, 3},
        {0, 1, 3}
};

std::vector<ov::test::utils::OpType> op_types = {
        ov::test::utils::OpType::SCALAR,
        ov::test::utils::OpType::VECTOR,
};

const std::vector<ov::test::utils::ReductionType> reduction_types = {
        ov::test::utils::ReductionType::Mean,
        ov::test::utils::ReductionType::Min,
        ov::test::utils::ReductionType::Max,
        ov::test::utils::ReductionType::Sum,
        ov::test::utils::ReductionType::Prod,
        ov::test::utils::ReductionType::L1,
        ov::test::utils::ReductionType::L2,
};

const std::vector<ov::test::utils::ReductionType> reduction_logical_types = {
        ov::test::utils::ReductionType::LogicalOr,
        ov::test::utils::ReductionType::LogicalAnd
};

const auto params_one_axis = testing::Combine(
        testing::Values(std::vector<int>{0}),
        testing::ValuesIn(op_types),
        testing::ValuesIn(keep_dims),
        testing::ValuesIn(reduction_types),
        testing::Values(model_types[0]),
        testing::ValuesIn(input_shapes_one_axis),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_one_axis_logical = testing::Combine(
        testing::Values(std::vector<int>{0}),
        testing::ValuesIn(op_types),
        testing::ValuesIn(keep_dims),
        testing::ValuesIn(reduction_logical_types),
        testing::Values(ov::element::boolean),
        testing::ValuesIn(input_shapes_one_axis),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_model_types = testing::Combine(
        testing::Values(std::vector<int>{1, 3}),
        testing::Values(op_types[1]),
        testing::ValuesIn(keep_dims),
        testing::Values(ov::test::utils::ReductionType::Max,
                        ov::test::utils::ReductionType::Mean,
                        ov::test::utils::ReductionType::Min,
                        ov::test::utils::ReductionType::Sum,
                        ov::test::utils::ReductionType::Prod),
        testing::ValuesIn(model_types),
        testing::Values(std::vector<size_t>{2, 2, 2, 2}),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_model_types_ReduceL1 = testing::Combine(
        testing::Values(std::vector<int>{1, 3}),
        testing::Values(op_types[1]),
        testing::ValuesIn(keep_dims),
        testing::Values(ov::test::utils::ReductionType::L1),
        testing::Values(ov::element::f32,
                        ov::element::f16,
                        ov::element::i64,
                        ov::element::i32),
        testing::Values(std::vector<size_t>{2, 2, 2, 2}),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_model_types_ReduceL2 = testing::Combine(
        testing::Values(std::vector<int>{1, 3}),
        testing::Values(op_types[1]),
        testing::ValuesIn(keep_dims),
        testing::Values(ov::test::utils::ReductionType::L2),
        testing::Values(ov::element::f32,
                        ov::element::f16),
        testing::Values(std::vector<size_t>{2, 2, 2, 2}),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_input_shapes = testing::Combine(
        testing::Values(std::vector<int>{0}),
        testing::Values(op_types[1]),
        testing::ValuesIn(keep_dims),
        testing::Values(ov::test::utils::ReductionType::Mean),
        testing::Values(model_types[0]),
        testing::Values(std::vector<size_t>{3},
                        std::vector<size_t>{3, 5},
                        std::vector<size_t>{2, 4, 6},
                        std::vector<size_t>{2, 4, 6, 8},
                        std::vector<size_t>{2, 2, 2, 2, 2},
                        std::vector<size_t>{2, 2, 2, 2, 2, 2}),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_axes = testing::Combine(
        testing::ValuesIn(axes),
        testing::Values(op_types[1]),
        testing::ValuesIn(keep_dims),
        testing::Values(ov::test::utils::ReductionType::Mean),
        testing::Values(model_types[0]),
        testing::ValuesIn(input_shapes),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_reduction_types = testing::Combine(
        testing::Values(std::vector<int>{0, 1, 3}),
        testing::Values(op_types[1]),
        testing::ValuesIn(keep_dims),
        testing::ValuesIn(reduction_types),
        testing::Values(model_types[0]),
        testing::Values(std::vector<size_t>{2, 9, 2, 9}),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_empty_input = testing::Combine(
        testing::ValuesIn(axes_0_dim),
        testing::Values(op_types[1]),
        testing::ValuesIn(keep_dims),
        testing::ValuesIn(reduction_types),
        testing::Values(model_types[0]),
        testing::ValuesIn(input_shapes_0_dim),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_reduction_types_logical = testing::Combine(
        testing::Values(std::vector<int>{0, 1, 3}),
        testing::Values(op_types[1]),
        testing::ValuesIn(keep_dims),
        testing::ValuesIn(reduction_logical_types),
        testing::Values(ov::element::boolean),
        testing::Values(std::vector<size_t>{2, 9, 2, 9}),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto params_ReduceSum_accuracy = testing::Combine(
        testing::Values(std::vector<int>{0}),
        testing::Values(op_types[1]),
        testing::Values(true),
        testing::Values(ov::test::utils::ReductionType::Sum),
        testing::Values(ov::element::f32),
        testing::Values(std::vector<size_t>{1000000}),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceSum_Accuracy,
        ReduceOpsLayerTest,
        params_ReduceSum_accuracy,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceOneAxis,
        ReduceOpsLayerTest,
        params_one_axis,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceLogicalOneAxis,
        ReduceOpsLayerTest,
        params_one_axis_logical,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_Precisions,
        ReduceOpsLayerTest,
        params_model_types,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_Precisions_L1,
        ReduceOpsLayerTest,
        params_model_types_ReduceL1,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_Precisions_L2,
        ReduceOpsLayerTest,
        params_model_types_ReduceL2,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_InputShapes,
        ReduceOpsLayerTest,
        params_input_shapes,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_Axes,
        ReduceOpsLayerTest,
        params_axes,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_ReductionTypes,
        ReduceOpsLayerTest,
        params_reduction_types,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_ReductionTypes_EmptyTensor,
        ReduceOpsLayerTest,
        params_empty_input,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceLogical_ReductionTypes,
        ReduceOpsLayerTest,
        params_reduction_types_logical,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce,
        ReduceOpsLayerWithSpecificInputTest,
        testing::Combine(
                testing::ValuesIn(decltype(axes) {{0}, {1}}),
                testing::Values(op_types[1]),
                testing::Values(true),
                testing::Values(ov::test::utils::ReductionType::Sum),
                testing::Values(ov::element::f32,
                                ov::element::i32),
                testing::Values(std::vector<size_t> {2, 10}),
                testing::Values(ov::test::utils::DEVICE_CPU)),
        ReduceOpsLayerWithSpecificInputTest::getTestCaseName
);

}  // namespace
