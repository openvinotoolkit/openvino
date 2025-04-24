// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selu.hpp"

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, selu_basic_inference_f32_3D) {
    const auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto selu = make_shared<op::v0::Selu>(param, alpha, lambda);

    ASSERT_EQ(selu->get_element_type(), element::f32);
    ASSERT_EQ(selu->get_shape(), (Shape{1, 32, 32}));
}

TEST(type_prop, selu_basic_inference_f16_3D) {
    const auto param = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1, 32, 32});
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1});
    const auto lambda = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1});
    const auto selu = make_shared<op::v0::Selu>(param, alpha, lambda);

    ASSERT_EQ(selu->get_element_type(), element::f16);
    ASSERT_EQ(selu->get_shape(), (Shape{1, 32, 32}));
}

TEST(type_prop, selu_basic_inference_f32_5D) {
    const auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 135, 221, 31, 15});
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto selu = make_shared<op::v0::Selu>(param, alpha, lambda);

    ASSERT_EQ(selu->get_element_type(), element::f32);
    ASSERT_EQ(selu->get_shape(), (Shape{12, 135, 221, 31, 15}));
}

TEST(type_prop, selu_basic_inference_f16_5D) {
    const auto param = make_shared<ov::op::v0::Parameter>(element::f16, Shape{12, 135, 221, 31, 15});
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1});
    const auto lambda = make_shared<ov::op::v0::Parameter>(element::f16, Shape{1});
    const auto selu = make_shared<op::v0::Selu>(param, alpha, lambda);

    ASSERT_EQ(selu->get_element_type(), element::f16);
    ASSERT_EQ(selu->get_shape(), (Shape{12, 135, 221, 31, 15}));
}

TEST(type_prop, selu_incompatible_input_type_boolean) {
    // Invalid data input element type
    try {
        auto data = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1, 2, 3, 4});
        const auto alpha = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1});
        const auto lambda = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1});
        auto selu = make_shared<op::v0::Selu>(data, alpha, lambda);
        // Data input expected to be of numeric type
        FAIL() << "Invalid input type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element types must be floating-point"));
    } catch (...) {
        FAIL() << "Input type check failed for unexpected reason";
    }
}

TEST(type_prop, selu_incompatible_input_type_i32) {
    // Invalid data input element type
    try {
        auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2, 3, 4});
        const auto alpha = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
        const auto lambda = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
        auto selu = make_shared<op::v0::Selu>(data, alpha, lambda);
        // Data input expected to be of numeric type
        FAIL() << "Invalid input type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element types must be floating-point"));
    } catch (...) {
        FAIL() << "Input type check failed for unexpected reason";
    }
}

TEST(type_prop, selu_incompatible_input_type_u16) {
    // Invalid data input element type
    try {
        auto data = make_shared<ov::op::v0::Parameter>(element::u16, Shape{1, 2, 3, 4});
        const auto alpha = make_shared<ov::op::v0::Parameter>(element::u16, Shape{1});
        const auto lambda = make_shared<ov::op::v0::Parameter>(element::u16, Shape{1});
        auto selu = make_shared<op::v0::Selu>(data, alpha, lambda);
        // Data input expected to be of numeric type
        FAIL() << "Invalid input type not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element types must be floating-point"));
    } catch (...) {
        FAIL() << "Input type check failed for unexpected reason";
    }
}

TEST(type_prop, selu_incompatible_input_types) {
    // Invalid data input element type
    try {
        auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
        const auto alpha = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
        const auto lambda = make_shared<ov::op::v0::Parameter>(element::u16, Shape{1});
        auto selu = make_shared<op::v0::Selu>(data, alpha, lambda);
        // Data input expected to be of numeric type
        FAIL() << "Inavlid input types not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element types do not match"));
    } catch (...) {
        FAIL() << "Input type check failed for unexpected reason";
    }
}

TEST(type_prop, selu_dynamic_rank_input_shape_2D) {
    const PartialShape param_shape{Dimension::dynamic(), 10};
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});
    const auto lambda = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto op = std::make_shared<op::v0::Selu>(param, alpha, lambda);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 10}));
}

TEST(type_prop, selu_dynamic_rank_input_shape_3D) {
    const PartialShape param_shape{100, Dimension::dynamic(), 58};
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto op = std::make_shared<op::v0::Selu>(param, alpha, lambda);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{100, Dimension(), 58}));
}

TEST(type_prop, selu_dynamic_rank_input_shape_full) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto op = std::make_shared<op::v0::Selu>(param, alpha, lambda);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
