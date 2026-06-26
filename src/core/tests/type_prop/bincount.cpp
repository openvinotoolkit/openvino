// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bincount.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using namespace testing;

class TypePropBincountV17Test : public TypePropOpTest<op::v17::Bincount> {};

TEST_F(TypePropBincountV17Test, default_ctor_unweighted) {
    auto data = std::make_shared<Parameter>(element::i32, Shape{5});
    auto bc = make_op(data);

    EXPECT_EQ(bc->get_output_element_type(0), element::i64);
    EXPECT_EQ(bc->get_input_size(), 1);
    EXPECT_EQ(bc->get_output_size(), 1);
}

TEST_F(TypePropBincountV17Test, dynamic_output_shape_no_const) {
    auto data = std::make_shared<Parameter>(element::i32, PartialShape{10});
    auto bc = make_op(data, 0);

    EXPECT_EQ(bc->get_output_element_type(0), element::i64);
    EXPECT_TRUE(bc->get_output_partial_shape(0).rank().is_static());
    EXPECT_EQ(bc->get_output_partial_shape(0).rank().get_length(), 1);
}

TEST_F(TypePropBincountV17Test, static_output_shape_from_constant) {
    auto data = std::make_shared<Constant>(element::i32, Shape{5}, std::vector<int32_t>{0, 1, 2, 1, 3});
    auto bc = make_op(data, 0);

    EXPECT_EQ(bc->get_output_element_type(0), element::i64);
    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape{4});
}

TEST_F(TypePropBincountV17Test, minlength_larger_than_data) {
    auto data = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});
    auto bc = make_op(data, 10);

    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape{10});
}

TEST_F(TypePropBincountV17Test, empty_data) {
    auto data = std::make_shared<Constant>(element::i32, Shape{0}, std::vector<int32_t>{});
    auto bc = make_op(data, 5);

    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape{5});
}

TEST_F(TypePropBincountV17Test, weighted_output_type_f32) {
    auto data = std::make_shared<Parameter>(element::i32, PartialShape{10});
    auto weights = std::make_shared<Parameter>(element::f32, PartialShape{10});
    auto bc = make_op(data, weights, 0);

    EXPECT_EQ(bc->get_output_element_type(0), element::f32);
}

TEST_F(TypePropBincountV17Test, weighted_output_type_f64) {
    auto data = std::make_shared<Parameter>(element::i64, PartialShape{10});
    auto weights = std::make_shared<Parameter>(element::f64, PartialShape{10});
    auto bc = make_op(data, weights, 0);

    EXPECT_EQ(bc->get_output_element_type(0), element::f64);
}

TEST_F(TypePropBincountV17Test, i64_data) {
    auto data = std::make_shared<Parameter>(element::i64, PartialShape{7});
    auto bc = make_op(data, 3);

    EXPECT_EQ(bc->get_output_element_type(0), element::i64);
}

TEST_F(TypePropBincountV17Test, dynamic_rank_data) {
    auto data = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    auto bc = make_op(data);

    EXPECT_EQ(bc->get_output_element_type(0), element::i64);
}

TEST_F(TypePropBincountV17Test, invalid_data_type) {
    auto data = std::make_shared<Parameter>(element::f32, Shape{5});
    OV_EXPECT_THROW(make_op(data), NodeValidationFailure, HasSubstr("element types: i32, i64, u8, u16, u32, u64"));
}

TEST_F(TypePropBincountV17Test, invalid_minlength_negative) {
    auto data = std::make_shared<Parameter>(element::i32, Shape{5});
    OV_EXPECT_THROW(make_op(data, -1), NodeValidationFailure, HasSubstr("minlength must be non-negative"));
}

TEST_F(TypePropBincountV17Test, invalid_data_rank_2d) {
    auto data = std::make_shared<Parameter>(element::i32, Shape{2, 3});
    OV_EXPECT_THROW(make_op(data), NodeValidationFailure, HasSubstr("1-D tensor"));
}

TEST_F(TypePropBincountV17Test, invalid_weights_rank_2d) {
    auto data = std::make_shared<Parameter>(element::i32, Shape{5});
    auto weights = std::make_shared<Parameter>(element::f32, Shape{1, 5});
    OV_EXPECT_THROW(make_op(data, weights, 0), NodeValidationFailure, HasSubstr("'weights' input must be a 1-D tensor"));
}

TEST_F(TypePropBincountV17Test, invalid_weights_length) {
    auto data = std::make_shared<Parameter>(element::i32, PartialShape{5});
    auto weights = std::make_shared<Parameter>(element::f32, PartialShape{6});
    OV_EXPECT_THROW(make_op(data, weights, 0), NodeValidationFailure, HasSubstr("must have the same length"));
}

TEST_F(TypePropBincountV17Test, invalid_negative_constant_data) {
    auto data = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, -1, 2});
    OV_EXPECT_THROW(make_op(data, 0), ov::AssertFailure, HasSubstr("must be non-negative"));
}

TEST_F(TypePropBincountV17Test, minlength_attribute) {
    auto data = std::make_shared<Parameter>(element::i32, Shape{5});
    auto bc = make_op(data, 7);
    EXPECT_EQ(bc->get_minlength(), 7);
    bc->set_minlength(10);
    EXPECT_EQ(bc->get_minlength(), 10);
}
