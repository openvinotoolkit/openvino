// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/inverse.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace testing;

class TypePropInverseV14Test : public TypePropOpTest<ov::op::v14::Inverse> {};

TEST_F(TypePropInverseV14Test, default_ctor) {
    const auto data = ov::op::v0::Constant::create(ov::element::f64, ov::Shape{2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto op = make_op();
    op->set_arguments(ov::OutputVector{data});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), ov::element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({2, 2}));
}

TEST_F(TypePropInverseV14Test, input_f64_2x2_constant) {
    const auto data = ov::op::v0::Constant::create(ov::element::f64, ov::Shape{2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto op = make_op(data, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{2, 2}));
}

TEST_F(TypePropInverseV14Test, symbol_propagation) {
    auto input_shape = ov::PartialShape{2, 2};
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    set_shape_symbols(input_shape, {A, B});
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    const auto op = make_op(data, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (input_shape));
    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), get_shape_symbols(input_shape));
}

TEST_F(TypePropInverseV14Test, input_f32_static) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{10, 4, 4});
    const auto op = make_op(data, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{10, 4, 4}));
}

TEST_F(TypePropInverseV14Test, input_f32_static_batch_dynamic_matrix) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{12, -1, -1});
    const auto op = make_op(data, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{12, -1, -1}));
}

TEST_F(TypePropInverseV14Test, input_f32_static_rank) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    const auto op = make_op(data, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{-1, -1, -1}));
}

TEST_F(TypePropInverseV14Test, input_f16_dynamic) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic());
    const auto op = make_op(data, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape::dynamic()));
}

TEST_F(TypePropInverseV14Test, input_f16_inverval_dimensions) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16,
        ov::PartialShape({ov::Dimension(10), ov::Dimension(2, 5), ov::Dimension(2, 5)}));
    const auto op = make_op(data, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (ov::PartialShape({ov::Dimension(10), ov::Dimension(2, 5), ov::Dimension(2, 5)})));
}

TEST_F(TypePropInverseV14Test, input_non_square) {
    const auto data = ov::op::v0::Constant::create(ov::element::f64, ov::Shape{2, 1}, {1.0f, 1.0f});
    OV_EXPECT_THROW(std::ignore = make_op(data, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Input must contain square matrices of the same shape."));
}

TEST_F(TypePropInverseV14Test, input_incompatibile_dims_1D) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4});
    OV_EXPECT_THROW(std::ignore = make_op(data, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Input must be at least a 2D matrix."));
}

TEST_F(TypePropInverseV14Test, input_incompatibile_dims_0D) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{});
    OV_EXPECT_THROW(std::ignore = make_op(data, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Input must be at least a 2D matrix."));
}

TEST_F(TypePropInverseV14Test, input_incompatibile_data_dtype) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4, 4});
    OV_EXPECT_THROW(std::ignore = make_op(data, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Expected floating point type as element type for the 'data' input."));
}
