// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/inverse.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace testing;

class TypePropInverseV14Test : public TypePropOpTest<ov::op::v14::Inverse> {};

TEST_F(TypePropInverseV14Test, input_f64_10x2x2) {
    const auto input = ov::op::v0::Constant::create(ov::element::f64, ov::Shape{2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto op = make_op(input, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{10, 3, 3}));
}

TEST_F(TypePropInverseV14Test, input_f32_static_rank_dynamic_dims) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{10, 4, 4});
    const auto op = make_op(input, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{-1, -1, -1}));
}

TEST_F(TypePropInverseV14Test, input_f16_fully_dynamic) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic());
    const auto op = make_op(input, false);
    EXPECT_EQ(op->get_element_type(), ov::element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape::dynamic()));
}

TEST_F(TypePropInverseV14Test, input_non_square) {
    const auto input = ov::op::v0::Constant::create(ov::element::f64, ov::Shape{2, 1}, {1.0f, 1.0f});
    OV_EXPECT_THROW(std::ignore = make_op(input, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Input must contain square matrices of the same shape."));
}

TEST_F(TypePropInverseV14Test, input_incompatibile_dims_1D) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4});
    OV_EXPECT_THROW(std::ignore = make_op(input, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Input must be at least a 2D matrix."));
}

TEST_F(TypePropInverseV14Test, input_incompatibile_dims_0D) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{});
    OV_EXPECT_THROW(std::ignore = make_op(input, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Input must be at least a 2D matrix."));
}

TEST_F(TypePropInverseV14Test, input_incompatibile_data_type) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    OV_EXPECT_THROW(std::ignore = make_op(input, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Expected float type as element type for the 'input' input."));
}
