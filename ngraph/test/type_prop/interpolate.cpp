//*****************************************************************************
// Copyright 2020 Intel Corporation
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

TEST(type_prop, interpolate_v4)
{
    using InterpolateMode = op::v4::Interpolate::InterpolateMode;
    using CoordinateTransformMode = op::v4::Interpolate::CoordinateTransformMode;
    using Nearest_mode = op::v4::Interpolate::NearestMode;
    using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;

    auto image = make_shared<op::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {15, 30});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::nearest;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto op = make_shared<op::v4::Interpolate>(image, output_shape, axes, attrs);

//     EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2, 15, 30}));
//     auto dyn_output_shape = make_shared<op::Parameter>(element::i64, Shape{2});
//     auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
//     auto mish_func = make_shared<op::v4::Mish>(data);
//     EXPECT_EQ(mish_func->get_element_type(), element::f32);
//     EXPECT_EQ(mish_func->get_shape(), (Shape{1, 3, 6}));
}


// TEST(type_prop_layers, interpolate_v4)
// {
//     auto image = make_shared<op::Parameter>(element::f32, Shape{2, 2, 33, 65});
//     auto dyn_output_shape = make_shared<op::Parameter>(element::i64, Shape{2});
//     auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {15, 30});
//     auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});
//     ASSERT_EQ(op->get_shape(), (Shape{2, 2, 15, 30}));
//
//     EXPECT_TRUE(make_shared<Interpolate>(image, dyn_output_shape, attrs)
//                     ->get_output_partial_shape(0)
//                     .same_scheme(PartialShape{2, 2, Dimension::dynamic(),
//                     Dimension::dynamic()}));
// }
