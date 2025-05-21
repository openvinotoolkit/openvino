// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::test {
using testing::HasSubstr;

class TypePropSparseFillEmptyRowsTest : public TypePropOpTest<op::v16::SparseFillEmptyRows> {};

TEST_F(TypePropSparseFillEmptyRowsTest, default_ctor_valid_inputs) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_element_type(2), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, with_const_inputs) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2});
    const auto dense_shape = std::make_shared<op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{3, 3});
    const auto indices =
        std::make_shared<op::v0::Constant>(element::i32, Shape{2, 2}, std::vector<int32_t>{0, 0, 2, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_element_type(2), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{3}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{3}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, dynamic_shapes) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_element_type(2), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, partially_dynamic_shapes) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_element_type(2), element::boolean);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, known_dense_shape) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    const auto dense_shape = std::make_shared<op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{5, 4});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic(), 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{5}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, known_values) {
    const auto values =
        std::make_shared<op::v0::Constant>(element::f32, Shape{3}, std::vector<float>{1.0f, 2.0f, 3.0f});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, known_indices) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices =
        std::make_shared<op::v0::Constant>(element::i32, Shape{3, 2}, std::vector<int32_t>{0, 0, 1, 0, 2, 0});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, known_default_value) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Constant>(element::f32, Shape{}, std::vector<float>{0.0f});

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, dense_shape_from_graph_shapeof) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto some_subgraph_result = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{5, 4});
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(some_subgraph_result, element::i32);
    const auto indices =
        std::make_shared<op::v0::Constant>(element::i32, Shape{3, 2}, std::vector<int32_t>{0, 0, 1, 0, 4, 0});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    const auto op = make_op(values, shape_of, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{5, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{5}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{5}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, symbol_propagation) {
    PartialShape values_shape{Dimension::dynamic()};
    PartialShape dense_shape_shape{2};
    PartialShape indices_shape{Dimension::dynamic(), 2};
    auto indices_symbols = set_shape_symbols(indices_shape);

    PartialShape default_value_shape{};
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, values_shape);
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, dense_shape_shape);
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, indices_shape);
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, default_value_shape);

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{Dimension::dynamic()}));

    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), testing::ElementsAre(nullptr, nullptr));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), testing::ElementsAre(nullptr));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(2)), testing::ElementsAre(nullptr));
}

TEST_F(TypePropSparseFillEmptyRowsTest, interval_shapes) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{{3, 6}});
    const auto dense_shape = std::make_shared<op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{8, 4});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{{3, 6}, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    const auto op = make_op(values, dense_shape, indices, default_value);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension::dynamic()}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{8}));
}

TEST_F(TypePropSparseFillEmptyRowsTest, invalid_values_rank) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The values input must be a 1D tensor"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, invalid_dense_shape_rank) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2, 2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The dense_shape input must be 1D"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, invalid_indices_rank) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The indices input must be a 2D tensor"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, invalid_default_value_rank) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The default_value input must be a scalar"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, invalid_indices_dimension) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 3});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("the second dimension having 2 elements"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, invalid_indices_element_type) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The element types of the dense_shape and indices inputs must match"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, invalid_dense_shape_element_type) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The element types of the dense_shape and indices inputs must match"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, invalid_merged_type) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::u32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::u32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The element type of the indices and dense_shape inputs must be i32 or i64"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, incompatible_indices_types) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("The element types of the dense_shape and indices inputs must match"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, mismatch_values_and_indices_dimensions) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4});
    const auto dense_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("first dimension matching the size of values"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, incorrect_dense_shape_dimension) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{3, 4, 5});
    const auto indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 2});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("have exactly 2 elements"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, out_of_bounds_row_index) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{3, 4});
    const auto indices =
        std::make_shared<op::v0::Constant>(element::i32, Shape{3, 2}, std::vector<int32_t>{0, 0, 1, 0, 3, 0});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("Sparse tensor index out of bounds: row 3 is outside the valid range [0, 2]"));
}

TEST_F(TypePropSparseFillEmptyRowsTest, out_of_bounds_column_index) {
    const auto values = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});
    const auto dense_shape = std::make_shared<op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{3, 4});
    const auto indices =
        std::make_shared<op::v0::Constant>(element::i32, Shape{3, 2}, std::vector<int32_t>{0, 0, 1, 4, 2, 0});
    const auto default_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(std::ignore = make_op(values, dense_shape, indices, default_value),
                    ov::NodeValidationFailure,
                    HasSubstr("Sparse tensor index out of bounds: column 4 is outside the valid range [0, 3]"));
}

}  // namespace ov::test
