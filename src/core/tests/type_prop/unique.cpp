// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <array>

#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;

namespace {
constexpr const size_t NUM_OUTPUTS = 4u;

void CHECK_OUTPUT_SHAPES(const std::shared_ptr<ov::Node>& op, std::array<PartialShape, NUM_OUTPUTS> expected_shapes) {
    for (size_t i = 0; i < NUM_OUTPUTS; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), expected_shapes[i])
            << "The output shape " << i << " of Unique is incorrect. Expected: " << expected_shapes[i]
            << ". Got: " << op->get_output_partial_shape(i);
    }
}

void CHECK_ELEMENT_TYPES(const std::shared_ptr<ov::Node>& op, std::array<element::Type, NUM_OUTPUTS> expected_types) {
    for (size_t i = 0; i < NUM_OUTPUTS; ++i) {
        EXPECT_EQ(op->get_output_element_type(i), expected_types[i])
            << "The output element type " << i << " of Unique is incorrect. Expected: " << expected_types[i]
            << ". Got: " << op->get_output_element_type(i);
    }
}
}  // namespace

TEST(type_prop, unique_no_axis_3d) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{2, 4, 2});
    const auto unique = make_shared<opset10::Unique>(data);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique,
                        {{PartialShape{{1, 16}}, PartialShape{{1, 16}}, PartialShape{{16}}, PartialShape{{1, 16}}}});
}

TEST(type_prop, unique_no_axis_3d_index_type_i32) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 3});
    const auto unique = make_shared<opset10::Unique>(data, true, element::i32);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i32, element::i32, element::i64}});
    CHECK_OUTPUT_SHAPES(unique,
                        {{PartialShape{{1, 9}}, PartialShape{{1, 9}}, PartialShape{{9}}, PartialShape{{1, 9}}}});
}

TEST(type_prop, unique_no_axis_3d_count_type_i32) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 3});
    const auto unique = make_shared<opset10::Unique>(data, true, element::i64, element::i32);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i32}});
    CHECK_OUTPUT_SHAPES(unique,
                        {{PartialShape{{1, 9}}, PartialShape{{1, 9}}, PartialShape{{9}}, PartialShape{{1, 9}}}});
}

TEST(type_prop, unique_no_axis_3d_all_outputs_i32) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 3});
    const auto unique = make_shared<opset10::Unique>(data, true, element::i32, element::i32);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i32, element::i32, element::i32}});
    CHECK_OUTPUT_SHAPES(unique,
                        {{PartialShape{{1, 9}}, PartialShape{{1, 9}}, PartialShape{{9}}, PartialShape{{1, 9}}}});
}

TEST(type_prop, unique_no_axis_scalar) {
    const auto data = make_shared<opset10::Parameter>(element::i32, PartialShape{});
    const auto unique = make_shared<opset10::Unique>(data);

    CHECK_ELEMENT_TYPES(unique, {{element::i32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique, {{PartialShape{1}, PartialShape{1}, PartialShape{1}, PartialShape{1}}});
}

TEST(type_prop, unique_no_axis_1D) {
    const auto data = make_shared<opset10::Parameter>(element::i64, PartialShape{1});
    const auto unique = make_shared<opset10::Unique>(data);

    CHECK_ELEMENT_TYPES(unique, {{element::i64, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique, {{PartialShape{1}, PartialShape{1}, PartialShape{1}, PartialShape{1}}});
}

TEST(type_prop, unique_3d_scalar_axis) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{2, 4, 2});
    const auto axis = make_shared<opset10::Constant>(element::i32, Shape{}, 1);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(
        unique,
        {{PartialShape{{2}, {1, 4}, {2}}, PartialShape{{1, 16}}, PartialShape{{4}}, PartialShape{{1, 16}}}});
}

TEST(type_prop, unique_3d_axis_1d) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{2, 4, 2});
    const auto axis = make_shared<opset10::Constant>(element::i32, Shape{1}, 2);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(
        unique,
        {{PartialShape{{2}, {4}, {1, 2}}, PartialShape{{1, 16}}, PartialShape{{2}}, PartialShape{{1, 16}}}});
}

TEST(type_prop, unique_3d_negative_axis) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{2, 4, 2});
    const auto axis = make_shared<opset10::Constant>(element::i64, Shape{}, -3);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(
        unique,
        {{PartialShape{{1, 2}, {4}, {2}}, PartialShape{{1, 16}}, PartialShape{{2}}, PartialShape{{1, 16}}}});
}

TEST(type_prop, unique_dynamic_dim_at_axis) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{2, -1, 2});
    const auto axis = make_shared<opset10::Constant>(element::i64, Shape{}, 1);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique,
                        {{PartialShape{{2}, {-1}, {2}}, PartialShape{{-1}}, PartialShape{{-1}}, PartialShape{{-1}}}});
}

TEST(type_prop, unique_dim_with_intervals_at_axis) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape{2, Dimension{2, 10}, 2});
    const auto axis = make_shared<opset10::Constant>(element::i64, Shape{}, 1);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(
        unique,
        {{PartialShape{{2}, {1, 10}, {2}}, PartialShape{{-1}}, PartialShape{{2, 10}}, PartialShape{{-1}}}});
}

TEST(type_prop, unique_dynamic_rank) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = make_shared<opset10::Constant>(element::i32, Shape{}, 1);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique,
                        {{PartialShape::dynamic(), PartialShape{{-1}}, PartialShape{{-1}}, PartialShape{{-1}}}});
}

TEST(type_prop, unique_all_dynamic_dims) {
    const auto data = make_shared<opset10::Parameter>(element::u8, PartialShape::dynamic(4));
    const auto axis = make_shared<opset10::Constant>(element::i32, Shape{}, -2);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_ELEMENT_TYPES(unique, {{element::u8, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique,
                        {{PartialShape::dynamic(4), PartialShape{{-1}}, PartialShape{{-1}}, PartialShape{{-1}}}});
}

TEST(type_prop, unique_all_dynamic_dims_no_axis) {
    const auto data = make_shared<opset10::Parameter>(element::u8, PartialShape::dynamic(4));
    const auto unique = make_shared<opset10::Unique>(data);

    CHECK_ELEMENT_TYPES(unique, {{element::u8, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique, {{PartialShape{{-1}}, PartialShape{{-1}}, PartialShape{{-1}}, PartialShape{{-1}}}});
}

TEST(type_prop, unique_dynamic_rank_negative_axis) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = make_shared<opset10::Constant>(element::i32, Shape{}, -1);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique,
                        {{PartialShape::dynamic(), PartialShape{{-1}}, PartialShape{{-1}}, PartialShape{{-1}}}});
}

TEST(type_prop, unique_dynamic_rank_no_axis) {
    const auto data = make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic());
    const auto unique = make_shared<opset10::Unique>(data);

    CHECK_ELEMENT_TYPES(unique, {{element::f32, element::i64, element::i64, element::i64}});
    CHECK_OUTPUT_SHAPES(unique, {{PartialShape{{-1}}, PartialShape{{-1}}, PartialShape{{-1}}, PartialShape{{-1}}}});
}

TEST(type_prop, unique_unsupported_index_et) {
    const auto data = make_shared<opset10::Parameter>(element::i64, PartialShape{1, 3, 3});
    EXPECT_THROW(const auto unique = make_shared<opset10::Unique>(data, true, element::f32), ov::NodeValidationFailure);
}

TEST(type_prop, unique_unsupported_axis_et) {
    const auto data = make_shared<opset10::Parameter>(element::i64, PartialShape{1, 3, 3});
    const auto axis = make_shared<opset10::Constant>(element::f32, Shape{}, 1.0f);

    EXPECT_THROW(const auto unique = make_shared<opset10::Unique>(data, axis), ov::NodeValidationFailure);
}

TEST(type_prop, unique_unsupported_axis_shape) {
    const auto data = make_shared<opset10::Parameter>(element::i64, PartialShape{1, 3, 3});
    const auto axis = make_shared<opset10::Constant>(element::i32, Shape{3, 3}, 1);

    EXPECT_THROW(const auto unique = make_shared<opset10::Unique>(data, axis), ov::NodeValidationFailure);
}

TEST(type_prop, unique_non_const_axis_input) {
    const auto data = make_shared<opset10::Parameter>(element::i64, PartialShape{1, 3, 3});
    const auto axis = make_shared<opset10::Parameter>(element::i32, Shape{});

    EXPECT_THROW(const auto unique = make_shared<opset10::Unique>(data, axis), ov::NodeValidationFailure);
}

TEST(type_prop, unique_with_zero_dimension) {
    const auto data = make_shared<opset10::Parameter>(element::i64, PartialShape{1, 0, 2});
    const auto axis = make_shared<opset10::Constant>(element::i32, Shape{}, 1);
    const auto unique = make_shared<opset10::Unique>(data, axis);

    CHECK_OUTPUT_SHAPES(unique, {{PartialShape{{1, 0, 2}}, PartialShape{{0}}, PartialShape{{0}}, PartialShape{{0}}}});
}

TEST(type_prop, unique_with_constant_input_no_axis) {
    const auto data = opset10::Constant::create(element::i32, Shape{5}, {5, 1, 4, 2, 5});
    const auto unique = make_shared<opset10::Unique>(data);

    CHECK_OUTPUT_SHAPES(unique, {{Shape{4}, Shape{4}, Shape{5}, Shape{4}}});
}
