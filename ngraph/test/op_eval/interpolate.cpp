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

#include "ngraph/op/constant.hpp"
#include "ngraph/op/interpolate.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
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

    std::vector<std::vector<int64_t>> spatial_shapes = {{3, 3}, {3, 3}, {8, 8}, {8, 8}, {8, 8}};

    std::vector<Shape> out_shapes = {
        Shape{1, 1, 3, 3}, Shape{1, 1, 3, 3}, Shape{1, 1, 8, 8}, Shape{1, 1, 8, 8},
        Shape{1, 1, 8, 8}};

    std::vector<std::vector<float>> scales_data = {
        {0.8f, 0.8f}, {0.8f, 0.8f}, {2.0f, 2.0f}, {2.0f, 2.0f}, {2.0f, 2.0f}};

    std::vector<float> cubic_coeffs = {-0.75f, -0.75f, -0.75f, -0.75f, -0.75f};

    std::vector<CoordinateTransformMode> transform_modes = {
        CoordinateTransformMode::half_pixel, CoordinateTransformMode::align_corners,
        CoordinateTransformMode::half_pixel, CoordinateTransformMode::align_corners,
        CoordinateTransformMode::asymmetric};

    std::vector<float> input_data = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    std::vector<int64_t> interp_axes = {2, 3};

    std::vector<std::vector<float>> expected_results = {
        {1.63078704f, 3.00462963f, 4.37847222f, 7.12615741f, 8.5f, 9.87384259f, 12.62152778f, 13.99537037f, 15.36921296f},
        {1.0f, 2.39519159f, 3.79038317f, 6.58076634f, 7.97595793f, 9.37114951f, 12.16153268f, 13.55672427f, 14.95191585f},
        {0.47265625f, 0.76953125f, 1.24609375f, 1.875f, 2.28125f, 2.91015625f, 3.38671875f, 3.68359375f,
         1.66015625f, 1.95703125f, 2.43359375f, 3.0625f, 3.46875f, 4.09765625f, 4.57421875f, 4.87109375f,
         3.56640625f, 3.86328125f, 4.33984375f, 4.96875f, 5.375f, 6.00390625f, 6.48046875f, 6.77734375f,
         6.08203125f, 6.37890625f, 6.85546875f, 7.484375f, 7.890625f, 8.51953125f, 8.99609375f, 9.29296875f,
         7.70703125f, 8.00390625f, 8.48046875f,  9.109375f, 9.515625f, 10.14453125f, 10.62109375f, 10.91796875f,
         10.22265625f, 10.51953125f, 10.99609375f, 11.625f, 12.03125f, 12.66015625f, 13.13671875f, 13.43359375f,
         12.12890625f, 12.42578125f, 12.90234375f, 13.53125f, 13.9375f, 14.56640625f, 15.04296875f, 15.33984375f,
         13.31640625f, 13.61328125f, 14.08984375f, 14.71875f, 15.125f, 15.75390625f, 16.23046875f, 16.52734375f},
        {1.0f, 1.34110787f, 1.80029155f, 2.32944606f, 2.67055394f, 3.19970845f, 3.65889213f, 4.0f,
         2.36443149f, 2.70553936f, 3.16472303f, 3.69387755f, 4.03498542f, 4.56413994f, 5.02332362f, 5.36443149f,
         4.20116618f, 4.54227405f, 5.00145773f, 5.53061224f, 5.87172012f, 6.40087464f, 6.86005831f, 7.20116618f,
         6.31778426f, 6.65889213f, 7.1180758f, 7.64723032f, 7.98833819f, 8.51749271f, 8.97667638f, 9.31778426f,
         7.68221574f, 8.02332362f, 8.48250729f, 9.01166181f, 9.35276968f, 9.8819242f, 10.34110787f, 10.68221574f,
         9.79883382f, 10.13994169f, 10.59912536f, 11.12827988f, 11.46938776f, 11.99854227f, 12.45772595f, 12.79883382f,
         11.63556851f, 11.97667638f, 12.43586006f, 12.96501458f, 13.30612245f, 13.83527697f, 14.29446064f, 14.63556851f,
         13.0f, 13.34110787f, 13.80029155f, 14.32944606f, 14.67055394f, 15.19970845f, 15.65889213f, 16.0f},
         {1.0f, 1.40625f, 2.0f, 2.5f, 3.0f, 3.59375f, 4.0f, 4.09375f, 2.625f, 3.03125f, 3.625f, 4.125f, 4.625f,
          5.21875f, 5.625f, 5.71875f, 5.0f, 5.40625f, 6.0f, 6.5f, 7.0f, 7.59375f, 8.0f, 8.09375f, 7.0f,
          7.40625f, 8.0f, 8.5f, 9.0f, 9.59375f, 10.0f, 10.09375f, 9.0f, 9.40625f, 10.0f, 10.5f, 11.0f, 11.59375f,
          12.0f, 12.09375f, 11.375f, 11.78125f, 12.375f, 12.875f, 13.375f, 13.96875f, 14.375f, 14.46875f, 13.0f,
          13.40625f, 14.0f, 14.5f, 15.0f, 15.59375f, 16.0f, 16.09375f, 13.375f, 13.78125f, 14.375f, 14.875f, 15.375f,
          15.96875f, 16.375f, 16.46875}};

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

        auto result = std::make_shared<HostTensor>();
        // ASSERT_TRUE(
        //     fun->evaluate({result},
        //                   {make_host_tensor<element::Type_t::f32>(data_shape, input_data),
        //                    make_host_tensor<element::Type_t::i64>(Shape{2}, spatial_shapes[i]),
        //                    make_host_tensor<element::Type_t::f32>(Shape{2}, scales_data[i]),
        //                    make_host_tensor<element::Type_t::i64>(Shape{2}, interp_axes)}));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), out_shapes[i]);
        // ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_results[i]));
    }
}
