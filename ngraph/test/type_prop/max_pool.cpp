//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, max_pool_auto_padding)
{
    const PartialShape arg_shape{1, 3, 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(
        arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({1, 3, 32, 32}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{1, 1}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_auto_padding_nc_dims_dynamic_same_lower)
{
    const PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(
        arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(
        {Dimension::dynamic(), Dimension::dynamic(), 32, 32}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{1, 1}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_auto_padding_nc_dims_dynamic_same_upper)
{
    const PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(
        arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(
        {Dimension::dynamic(), Dimension::dynamic(), 32, 32}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{0, 0}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{1, 1}));
}

TEST(type_prop, max_pool_auto_padding_spatial_dims_dynamic)
{
    const PartialShape arg_shape{1, 3, 32, Dimension::dynamic()};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(
        arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({1, 3, 32, Dimension::dynamic()}));
    ASSERT_EQ(mp->get_pads_begin(), (Shape{1, 0}));
    ASSERT_EQ(mp->get_pads_end(), (Shape{0, 0}));
}

TEST(type_prop, max_pool_default_values)
{
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
