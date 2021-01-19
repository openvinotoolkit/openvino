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

using namespace ngraph;

using InterpolateMode = op::v4::Interpolate::InterpolateMode;
using CoordinateTransformMode = op::v4::Interpolate::CoordinateTransformMode;
using Nearest_mode = op::v4::Interpolate::NearestMode;
using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;
using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;

TEST(type_prop, interpolate_v4)
{
    auto image = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto target_shape = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::nearest;
    attrs.shape_calculation_mode = ShapeCalcMode::scales;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{2, 2, 15, 30}));
}

TEST(type_prop, interpolate_v4_partial)
{
    auto partial_shape = PartialShape{2, 2, Dimension::dynamic(), Dimension::dynamic()};

    auto image = std::make_shared<op::Parameter>(element::f32, partial_shape);
    auto target_shape = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::nearest;
    attrs.shape_calculation_mode = ShapeCalcMode::scales;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(partial_shape));

    // rank unknown
    auto partial_param = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto interp_part =
        std::make_shared<op::v4::Interpolate>(partial_param, target_shape, scales, axes, attrs);
    ASSERT_TRUE(interp_part->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, interpolate_v4_partial_static_rank)
{
    auto partial_shape = PartialShape{2, 2, Dimension::dynamic(), Dimension::dynamic()};

    auto image = std::make_shared<op::Parameter>(element::f32, partial_shape);
    auto target_shape = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::nearest;
    attrs.shape_calculation_mode = ShapeCalcMode::scales;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(partial_shape));
    ASSERT_TRUE(interp->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, interpolate_v4_partial_static_rank2)
{
    auto partial_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 10, 20};
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 5, 10};

    auto image = std::make_shared<op::Parameter>(element::f32, partial_shape);
    auto target_shape = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::nearest;
    attrs.shape_calculation_mode = ShapeCalcMode::scales;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(out_shape));
    ASSERT_TRUE(interp->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, interpolate_v4_partial_static_rank3)
{
    auto partial_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3};
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1, 1};

    auto image = std::make_shared<op::Parameter>(element::f32, partial_shape);
    auto target_shape = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {1.0f / 3.0f, 1.0f / 3.0f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::nearest;
    attrs.shape_calculation_mode = ShapeCalcMode::scales;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(out_shape));
    ASSERT_TRUE(interp->get_output_partial_shape(0).rank().is_static());
}
