// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

// ------------------------------ V0 ------------------------------
TEST(type_prop, gelu_v0) {
    const PartialShape param_shape{64, Dimension::dynamic(), 256, Dimension(4, 8)};
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<op::v0::Gelu>(param);
    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_output_partial_shape(0), param_shape);
}

// ------------------------------ V7 ------------------------------
TEST(type_prop, gelu_default_mode_inference_f32) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    auto gelu = make_shared<op::v7::Gelu>(param);

    ASSERT_EQ(gelu->get_element_type(), element::f32);
    ASSERT_EQ(gelu->get_shape(), (Shape{1, 32, 32}));
    ASSERT_EQ(gelu->get_approximation_mode(), op::GeluApproximationMode::ERF);
}

TEST(type_prop, gelu_default_mode_inference_f16) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1, 32, 32});
    auto gelu = make_shared<op::v7::Gelu>(param);

    ASSERT_EQ(gelu->get_element_type(), element::f16);
    ASSERT_EQ(gelu->get_shape(), (Shape{1, 32, 32}));
    ASSERT_EQ(gelu->get_approximation_mode(), op::GeluApproximationMode::ERF);
}

TEST(type_prop, gelu_tanh_mode_inference_f32) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    auto gelu = make_shared<op::v7::Gelu>(param, op::GeluApproximationMode::TANH);

    ASSERT_EQ(gelu->get_element_type(), element::f32);
    ASSERT_EQ(gelu->get_shape(), (Shape{1, 32, 32}));
    ASSERT_EQ(gelu->get_approximation_mode(), op::GeluApproximationMode::TANH);
}

TEST(type_prop, gelu_tanh_mode_inference_f16) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1, 32, 32});
    auto gelu = make_shared<op::v7::Gelu>(param, op::GeluApproximationMode::TANH);

    ASSERT_EQ(gelu->get_element_type(), element::f16);
    ASSERT_EQ(gelu->get_shape(), (Shape{1, 32, 32}));
    ASSERT_EQ(gelu->get_approximation_mode(), op::GeluApproximationMode::TANH);
}

TEST(type_prop, gelu_incompatible_input_type_boolean) {
    auto param = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1, 32, 32});
    ASSERT_THROW(const auto unused = std::make_shared<op::v7::Gelu>(param), ov::NodeValidationFailure);
}

TEST(type_prop, gelu_incompatible_input_type_u16) {
    auto param = make_shared<ov::op::v0::Parameter>(element::u16, Shape{1, 32, 32});
    ASSERT_THROW(const auto unused = std::make_shared<op::v7::Gelu>(param), ov::NodeValidationFailure);
}

TEST(type_prop, gelu_incompatible_input_type_i32) {
    auto param = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 32, 32});
    ASSERT_THROW(const auto unused = std::make_shared<op::v7::Gelu>(param), ov::NodeValidationFailure);
}

TEST(type_prop, gelu_incompatible_input_type_i16) {
    auto param = make_shared<ov::op::v0::Parameter>(element::i16, Shape{1, 32, 32});
    ASSERT_THROW(const auto unused = std::make_shared<op::v7::Gelu>(param), ov::NodeValidationFailure);
}

TEST(type_prop, gelu_dynamic_rank_input_shape_2D) {
    const PartialShape param_shape{Dimension::dynamic(), 10};
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<op::v7::Gelu>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 10}));
}

TEST(type_prop, gelu_dynamic_rank_input_shape_3D) {
    const PartialShape param_shape{100, Dimension::dynamic(), 58};
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    const auto op = std::make_shared<op::v7::Gelu>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{100, Dimension(), 58}));
}

TEST(type_prop, gelu_dynamic_rank_input_shape_full) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto op = std::make_shared<op::v7::Gelu>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
