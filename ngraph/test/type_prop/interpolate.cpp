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

    auto image = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::nearest;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
    attrs.nearest_mode = Nearest_mode::round_prefer_floor;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, scales, axes, attrs);

    EXPECT_EQ(interp->get_shape(), (Shape{2, 2, 15, 30}));
}
