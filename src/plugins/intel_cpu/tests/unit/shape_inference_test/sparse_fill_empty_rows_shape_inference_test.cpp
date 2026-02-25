// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"
#include "openvino/op/sparse_fill_empty_rows.hpp"
#include "openvino/op/constant.hpp"

using namespace ov::intel_cpu;
using ov::op::v0::Constant, ov::op::v0::Parameter;
using testing::HasSubstr;

struct SparseFillEmptyRowsTestParams {
    ov::Shape values_shape;
    ov::Shape dense_shape_shape;
    ov::Shape indices_shape;
    ov::Shape default_value_shape;
    std::vector<int32_t> dense_shape_val;
    std::vector<int32_t> indices_val;
    ov::Shape expected_output_indices_shape;
    ov::Shape expected_output_values_shape;
    ov::Shape expected_empty_row_indicator_shape;
};

class SparseFillEmptyRowsStaticShapeInferenceTest: public OpStaticShapeInferenceTest<ov::op::v16::SparseFillEmptyRows> {};

class SparseFillEmptyRowsStaticTestSuite : public SparseFillEmptyRowsStaticShapeInferenceTest,
                                         public ::testing::WithParamInterface<SparseFillEmptyRowsTestParams> {};

TEST_F(SparseFillEmptyRowsStaticShapeInferenceTest, input_from_tensor_accessor) {
    const auto values = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto dense_shape = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape::dynamic());
    const auto indices = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape::dynamic());
    const auto default_value = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto op = make_op(values, dense_shape, indices, default_value);

    float values_val[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t dense_shape_val[] = {5, 6};
    int32_t indices_val[] = {0, 1, 0, 3, 2, 0, 3, 1};
    float default_value_val[] = {0.0f};

    auto const_inputs = std::unordered_map<size_t, ov::Tensor>{
        {0, {ov::element::f32, ov::Shape{4}, values_val}},
        {1, {ov::element::i32, ov::Shape{2}, dense_shape_val}},
        {2, {ov::element::i32, ov::Shape{4, 2}, indices_val}},
        {3, {ov::element::f32, ov::Shape{}, default_value_val}}
    };

    const auto input_shapes = StaticShapeVector{{4}, {2}, {4, 2}, {}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor(const_inputs));

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({6, 2}));
    EXPECT_EQ(output_shapes[1], StaticShape({6}));
    EXPECT_EQ(output_shapes[2], StaticShape({5}));
}

TEST_F(SparseFillEmptyRowsStaticShapeInferenceTest, static_shapes) {
    const auto values = std::make_shared<Parameter>(ov::element::f32, ov::Shape{4});
    const auto dense_shape = std::make_shared<Parameter>(ov::element::i32, ov::Shape{2});
    const auto indices = std::make_shared<Parameter>(ov::element::i32, ov::Shape{4, 2});
    const auto default_value = std::make_shared<Parameter>(ov::element::f32, ov::Shape{});
    const auto op = make_op(values, dense_shape, indices, default_value);

    int32_t dense_shape_val[] = {8, 5};
    int32_t indices_val[] = {0, 1, 0, 3, 2, 0, 3, 1};
    auto const_inputs = std::unordered_map<size_t, ov::Tensor>{
        {1, {ov::element::i32, ov::Shape{2}, dense_shape_val}},
        {2, {ov::element::i32, ov::Shape{4, 2}, indices_val}}
    };

    const auto input_shapes = StaticShapeVector{{5}, {2}, {5, 2}, {}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor(const_inputs));

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape({10, 2}));
    EXPECT_EQ(output_shapes[1], StaticShape({10}));
    EXPECT_EQ(output_shapes[2], StaticShape({8}));
}

TEST_P(SparseFillEmptyRowsStaticTestSuite, sparse_fill_empty_rows_static_shape_inference) {
    const auto& [values_shape, dense_shape_shape, indices_shape, default_value_shape,
                dense_shape_val, indices_val,
                expected_output_indices_shape, expected_output_values_shape, expected_empty_row_indicator_shape] = GetParam();

    const auto values = std::make_shared<Parameter>(ov::element::f32, values_shape);
    const auto dense_shape = std::make_shared<Constant>(ov::element::i32, dense_shape_shape, dense_shape_val);
    const auto indices = std::make_shared<Constant>(ov::element::i32, indices_shape, indices_val);
    const auto default_value = std::make_shared<Parameter>(ov::element::f32, default_value_shape);

    const auto op = make_op(values, dense_shape, indices, default_value);

    const auto input_shapes = StaticShapeVector{values_shape, dense_shape_shape, indices_shape, default_value_shape};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, ov::make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes[0], StaticShape(expected_output_indices_shape));
    EXPECT_EQ(output_shapes[1], StaticShape(expected_output_values_shape));
    EXPECT_EQ(output_shapes[2], StaticShape(expected_empty_row_indicator_shape));
}

INSTANTIATE_TEST_SUITE_P(SparseFillEmptyRowsStaticShapeInferenceTests,
                         SparseFillEmptyRowsStaticTestSuite,
                         ::testing::Values(
    // No empty rows
    SparseFillEmptyRowsTestParams{
        ov::Shape{3},                               // values_shape
        ov::Shape{2},                               // dense_shape_shape
        ov::Shape{3, 2},                            // indices_shape
        ov::Shape{},                                // default_value_shape
        std::vector<int32_t>{3, 4},                 // dense_shape_val
        std::vector<int32_t>{0, 0, 1, 0, 2, 0},     // indices_val
        ov::Shape{3, 2},                            // expected_output_indices_shape
        ov::Shape{3},                               // expected_output_values_shape
        ov::Shape{3}                                // expected_empty_row_indicator_shape
    },
    // One empty row in the middle
    SparseFillEmptyRowsTestParams{
        ov::Shape{3},                               // values_shape
        ov::Shape{2},                               // dense_shape_shape
        ov::Shape{3, 2},                            // indices_shape
        ov::Shape{},                                // default_value_shape
        std::vector<int32_t>{4, 4},                 // dense_shape_val
        std::vector<int32_t>{0, 0, 1, 0, 3, 0},     // indices_val
        ov::Shape{4, 2},                            // expected_output_indices_shape
        ov::Shape{4},                               // expected_output_values_shape
        ov::Shape{4}                                // expected_empty_row_indicator_shape
    },
    // Multiple empty rows
    SparseFillEmptyRowsTestParams{
        ov::Shape{2},                               // values_shape
        ov::Shape{2},                               // dense_shape_shape
        ov::Shape{2, 2},                            // indices_shape
        ov::Shape{},                                // default_value_shape
        std::vector<int32_t>{5, 3},                 // dense_shape_val
        std::vector<int32_t>{0, 0, 4, 0},           // indices_val
        ov::Shape{5, 2},                            // expected_output_indices_shape
        ov::Shape{5},                               // expected_output_values_shape
        ov::Shape{5}                                // expected_empty_row_indicator_shape
    },
    // All rows empty
    SparseFillEmptyRowsTestParams{
        ov::Shape{0},                               // values_shape
        ov::Shape{2},                               // dense_shape_shape
        ov::Shape{0, 2},                            // indices_shape
        ov::Shape{},                                // default_value_shape
        std::vector<int32_t>{3, 4},                 // dense_shape_val
        std::vector<int32_t>{},                     // indices_val
        ov::Shape{3, 2},                            // expected_output_indices_shape
        ov::Shape{3},                               // expected_output_values_shape
        ov::Shape{3}                                // expected_empty_row_indicator_shape
    }
));
