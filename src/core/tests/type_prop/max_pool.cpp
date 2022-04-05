// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, max_pool_valid_auto_padding) {
    const PartialShape arg_shape{1, 3, 32};
    const Strides strides{1};
    const Shape pads_begin{2};
    const Shape pads_end{2};
    const Shape kernel_shape{2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::VALID;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({1, 3, 31}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{0}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0}));
}

TEST(type_prop, max_pool_1D_auto_padding) {
    const PartialShape arg_shape{1, 3, 32};
    const Strides strides{1};
    const Shape pads_begin{0};
    const Shape pads_end{0};
    const Shape kernel_shape{2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({1, 3, 32}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{1}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0}));
}

TEST(type_prop, max_pool_2D_auto_padding) {
    const PartialShape arg_shape{1, 3, 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({1, 3, 32, 32}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{1, 1}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_auto_padding_1D_nc_dims_dynamic_same_lower) {
    const PartialShape arg_shape{Dimension::dynamic(), 32, 32};
    const Strides strides{1};
    const Shape pads_begin{0};
    const Shape pads_end{0};
    const Shape kernel_shape{2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), 32, 32}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{1}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0}));
}

TEST(type_prop, max_pool_auto_padding_2D_nc_dims_dynamic_same_lower) {
    const PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), Dimension::dynamic(), 32, 32}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{1, 1}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_auto_padding_nc_dims_dynamic_same_upper) {
    const PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), Dimension::dynamic(), 32, 32}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{0, 0}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{1, 1}));
}

TEST(type_prop, max_pool_auto_padding_spatial_dims_dynamic) {
    const PartialShape arg_shape{1, 3, 32, Dimension::dynamic()};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({1, 3, 32, Dimension::dynamic()}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{1, 0}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_default_values) {
    const PartialShape arg_shape{1, 3, 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(arg, strides, pads_begin, pads_end, kernel_shape);

    ASSERT_EQ(mp->get_rounding_type(), op::RoundingType::FLOOR);
    ASSERT_EQ(mp->get_auto_pad(), op::PadType::EXPLICIT);
}

TEST(type_prop, max_pool_v8_3D_no_dilations) {
    const PartialShape arg_shape{1, 7, 13};
    const Strides strides{1};
    const Strides dilations{1};
    const Shape pads_begin{0};
    const Shape pads_end{0};
    const Shape kernel_shape{3};

    const auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    const auto mp = make_shared<op::v8::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 7, 11});
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(expected_output_shape));
    ASSERT_TRUE(mp->get_output_partial_shape(1).same_scheme(expected_output_shape));
}

TEST(type_prop, max_pool_v8_3D_with_dilations) {
    const PartialShape arg_shape{1, 7, 13};
    const Strides strides{1};
    const Strides dilations{2};
    const Shape pads_begin{0};
    const Shape pads_end{0};
    const Shape kernel_shape{3};

    const auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    const auto mp = make_shared<op::v8::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 7, 9});
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(expected_output_shape));
    ASSERT_TRUE(mp->get_output_partial_shape(1).same_scheme(expected_output_shape));
}

TEST(type_prop, max_pool_v8_3D_with_dilations_and_padding) {
    const PartialShape arg_shape{1, 7, 13};
    const Strides strides{1};
    const Strides dilations{2};
    const Shape pads_begin{1};
    const Shape pads_end{2};
    const Shape kernel_shape{3};

    const auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    const auto mp = make_shared<op::v8::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 7, 12});
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(expected_output_shape));
    ASSERT_TRUE(mp->get_output_partial_shape(1).same_scheme(expected_output_shape));
}

TEST(type_prop, max_pool_v8_4D_no_dilations) {
    const PartialShape arg_shape{1, 3, 13, 13};
    const Strides strides{1, 1};
    const Strides dilations{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};

    const auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    const auto mp = make_shared<op::v8::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 3, 12, 12});
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(expected_output_shape));
    ASSERT_TRUE(mp->get_output_partial_shape(1).same_scheme(expected_output_shape));
}

TEST(type_prop, max_pool_v8_4D_with_dilations) {
    const PartialShape arg_shape{1, 3, 13, 13};
    const Strides strides{1, 1};
    const Strides dilations{2, 3};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};

    const auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    const auto mp = make_shared<op::v8::MaxPool>(arg, strides, dilations, pads_begin, pads_end, kernel_shape);

    const auto expected_output_shape = PartialShape({1, 3, 11, 10});
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(expected_output_shape));
    ASSERT_TRUE(mp->get_output_partial_shape(1).same_scheme(expected_output_shape));
}
