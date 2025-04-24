// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/clamp.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

using namespace std;

TEST(type_prop, clamp_basic_f32) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32, 32});
    auto clamp = make_shared<ov::op::v0::Clamp>(data, 0.0, 2.1);

    ASSERT_EQ(clamp->get_element_type(), ov::element::f32);
    ASSERT_EQ(clamp->get_min(), 0.0);
    ASSERT_EQ(clamp->get_max(), 2.1);
    ASSERT_EQ(clamp->get_output_shape(0), (ov::Shape{1, 32, 32}));
}

TEST(type_prop, clamp_basic_i32) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 32, 32});
    auto clamp = make_shared<ov::op::v0::Clamp>(data, 0.0, 2.1);

    ASSERT_EQ(clamp->get_element_type(), ov::element::i32);
    ASSERT_EQ(clamp->get_min(), 0.0);
    ASSERT_EQ(clamp->get_max(), 2.1);
    ASSERT_EQ(clamp->get_output_shape(0), (ov::Shape{1, 32, 32}));
}

TEST(type_prop, clamp_shape_static_rank) {
    auto data =
        make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                           ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 32});
    auto clamp = make_shared<ov::op::v0::Clamp>(data, -2.1, 2.1);

    ASSERT_EQ(clamp->get_element_type(), ov::element::f16);
    ASSERT_EQ(clamp->get_min(), -2.1);
    ASSERT_EQ(clamp->get_max(), 2.1);
    ASSERT_EQ(clamp->get_output_partial_shape(0),
              (ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 32}));
}

TEST(type_prop, clamp_shape_dynamic) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::PartialShape::dynamic());
    auto clamp = make_shared<ov::op::v0::Clamp>(data, 1.5, 15.0);

    ASSERT_EQ(clamp->get_element_type(), ov::element::u16);
    ASSERT_EQ(clamp->get_min(), 1.5);
    ASSERT_EQ(clamp->get_max(), 15.0);
    ASSERT_EQ(clamp->get_output_partial_shape(0), (ov::PartialShape::dynamic()));
}

TEST(type_prop, clamp_evaluate_bounds) {
    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<ov::op::v3::ShapeOf>(param);
    auto gather = make_shared<ov::op::v1::Gather>(shape_of,
                                                  ov::op::v0::Constant::create(ov::element::i64, {3}, {2, 1, 0}),
                                                  ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto clamp = make_shared<ov::op::v0::Clamp>(gather, 0, 5);
    auto r = make_shared<ov::op::v1::Reshape>(param, clamp, false);

    ASSERT_EQ(r->get_element_type(), ov::element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), ov::PartialShape({3, 2, ov::Dimension(1, 5)}));
}

TEST(type_prop, clamp_invalid_element_type) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{2, 2});

    try {
        auto clamp = make_shared<ov::op::v0::Clamp>(data, 0.5, 5.5);
        // Input element type is boolean
        FAIL() << "Invalid boolean element type for input not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Input element type must be numeric");
    } catch (...) {
        FAIL() << "Numeric element type node validation check failed for unexpected reason";
    }
}

TEST(type_prop, clamp_equal_attributes) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{2, 2});

    auto clamp = make_shared<ov::op::v0::Clamp>(data, 1.0, 1.0);
    ASSERT_EQ(clamp->get_element_type(), ov::element::f64);
    ASSERT_EQ(clamp->get_min(), 1.0);
    ASSERT_EQ(clamp->get_max(), 1.0);
    ASSERT_EQ(clamp->get_output_shape(0), (ov::Shape{2, 2}));
}

TEST(type_prop, clamp_invalid_attributes) {
    auto data = make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{2, 2});

    try {
        auto clamp = make_shared<ov::op::v0::Clamp>(data, 2.0, 1.0);
        // Attribute 'max' not greater than 'min'
        FAIL() << "Attribute 'min' bigger than 'max' not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Attribute 'min' must be less or equal than 'max'");
    } catch (...) {
        FAIL() << "'min' and 'max' attributes node validation check failed for unexpected reason";
    }
}
