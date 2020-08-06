//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/interpolate.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using InterpolateMode = op::v4::Interpolate::InterpolateMode;
using CoordinateTransformMode = op::v4::Interpolate::CoordinateTransformMode;
using Nearest_mode = op::v4::Interpolate::NearestMode;
using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;
using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;

TEST(op_eval, interpolate_v4_scales_cubic)
{
    auto data_shape = Shape{1, 1, 4, 4};

    std::vector<std::vector<int64_t>> spatial_shapes = {{3, 3}};

    std::vector<std::vector<float>> scales_data = {{0.8f, 0.8f}};

    std::vector<float> cubic_coeffs = {-0.75f};

    std::vector<CoordinateTransformMode> transform_modes = {
        CoordinateTransformMode::half_pixel};

    std::vector<float> input_data = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    std::vector<int64_t> interp_axes = {2, 3};

    std::size_t num_of_tests = transform_modes.size();

    for (std::size_t i = 0; i < num_of_tests; ++i)
    {
        auto image = std::make_shared<op::Parameter>(element::f32, data_shape);
        auto target_spatial_shape = std::make_shared<op::Parameter>(element::i64, Shape{2});
        auto scales = std::make_shared<op::Parameter>(element::f32, Shape{2});
        auto axes = std::make_shared<op::Parameter>(element::i64, Shape{2});

        InterpolateAttrs attrs;
        attrs.mode = InterpolateMode::cubic;
        attrs.shape_calculation_mode = ShapeCalcMode::scales;
        attrs.coordinate_transformation_mode = transform_modes[i];
        attrs.nearest_mode = Nearest_mode::round_prefer_floor;
        attrs.antialias = false;
        attrs.pads_begin = {0, 0, 0, 0};
        attrs.pads_end = {0, 0, 0, 0};
        attrs.cube_coeff = cubic_coeffs[i];

        auto interp =
            std::make_shared<op::v4::Interpolate>(image, target_spatial_shape, scales, axes, attrs);
        auto fun = std::make_shared<Function>(
            OutputVector{interp}, ParameterVector{image, target_spatial_shape, scales, axes});

        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(
            fun->evaluate({result},
                          {make_host_tensor<element::Type_t::f32>(data_shape, input_data),
                           make_host_tensor<element::Type_t::i64>(Shape{2}, spatial_shapes[i]),
                           make_host_tensor<element::Type_t::f32>(Shape{2}, scales_data[i]),
                           make_host_tensor<element::Type_t::i64>(Shape{2}, interp_axes)}));
    }
}
