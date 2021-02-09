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

// All examples are from ONNX Resize-11 documentation
// (see https://github.com/onnx/onnx/blob/master/docs/Operators.md).
TEST(op_eval, interpolate_v4_cubic)
{
    auto data_shape = Shape{1, 1, 4, 4};

    struct ShapesAndAttrs
    {
        std::vector<int64_t> spatial_shape;
        Shape out_shape;
        std::vector<float> scales_data;
        CoordinateTransformMode transform_mode;
        ShapeCalcMode shape_calculation_mode;
    };

    std::vector<ShapesAndAttrs> shapes_and_attrs = {
        // resize_downsample_scales_cubic:
        ShapesAndAttrs{{3, 3},
                       Shape{1, 1, 3, 3},
                       {0.8f, 0.8f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales},
        // resize_downsample_sizes_cubic:
        ShapesAndAttrs{{3, 3},
                       Shape{1, 1, 3, 3},
                       {0.75f, 0.75f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::sizes},
        // resize_upsample_scales_cubic:
        ShapesAndAttrs{{8, 8},
                       Shape{1, 1, 8, 8},
                       {2.0f, 2.0f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales},
        // resize_upsample_scales_cubic_asymmetric:
        ShapesAndAttrs{{8, 8},
                       Shape{1, 1, 8, 8},
                       {2.0f, 2.0f},
                       CoordinateTransformMode::asymmetric,
                       ShapeCalcMode::scales},
        // resize_upsample_sizes_cubic:
        ShapesAndAttrs{{9, 10},
                       Shape{1, 1, 9, 10},
                       {2.25f, 2.5f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::sizes},
        // resize_downsample_scales_cubic_align_corners:
        // (expected values from ONNX documentation are incorrect!)
        ShapesAndAttrs{{3, 3},
                       Shape{1, 1, 3, 3},
                       {0.8f, 0.8f},
                       CoordinateTransformMode::align_corners,
                       ShapeCalcMode::scales},
        // resize_upsample_scales_cubic_align_corners:
        ShapesAndAttrs{{8, 8},
                       Shape{1, 1, 8, 8},
                       {2.0f, 2.0f},
                       CoordinateTransformMode::align_corners,
                       ShapeCalcMode::scales}};

    std::vector<float> input_data = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    std::vector<std::vector<float>> expected_results = {
        {1.47119141,
         2.78125,
         4.08251953,
         6.71142578,
         8.02148438,
         9.32275391,
         11.91650391,
         13.2265625,
         14.52783203},
        {1.63078704f,
         3.00462963f,
         4.37847222f,
         7.12615741f,
         8.5f,
         9.87384259f,
         12.62152778f,
         13.99537037f,
         15.36921296f},
        {0.47265625f,  0.76953125f,  1.24609375f,  1.875f,       2.28125f,     2.91015625f,
         3.38671875f,  3.68359375f,  1.66015625f,  1.95703125f,  2.43359375f,  3.0625f,
         3.46875f,     4.09765625f,  4.57421875f,  4.87109375f,  3.56640625f,  3.86328125f,
         4.33984375f,  4.96875f,     5.375f,       6.00390625f,  6.48046875f,  6.77734375f,
         6.08203125f,  6.37890625f,  6.85546875f,  7.484375f,    7.890625f,    8.51953125f,
         8.99609375f,  9.29296875f,  7.70703125f,  8.00390625f,  8.48046875f,  9.109375f,
         9.515625f,    10.14453125f, 10.62109375f, 10.91796875f, 10.22265625f, 10.51953125f,
         10.99609375f, 11.625f,      12.03125f,    12.66015625f, 13.13671875f, 13.43359375f,
         12.12890625f, 12.42578125f, 12.90234375f, 13.53125f,    13.9375f,     14.56640625f,
         15.04296875f, 15.33984375f, 13.31640625f, 13.61328125f, 14.08984375f, 14.71875f,
         15.125f,      15.75390625f, 16.23046875f, 16.52734375f},
        {1.0f,    1.40625f,  2.0f,    2.5f,    3.0f,    3.59375f,  4.0f,    4.09375f,
         2.625f,  3.03125f,  3.625f,  4.125f,  4.625f,  5.21875f,  5.625f,  5.71875f,
         5.0f,    5.40625f,  6.0f,    6.5f,    7.0f,    7.59375f,  8.0f,    8.09375f,
         7.0f,    7.40625f,  8.0f,    8.5f,    9.0f,    9.59375f,  10.0f,   10.09375f,
         9.0f,    9.40625f,  10.0f,   10.5f,   11.0f,   11.59375f, 12.0f,   12.09375f,
         11.375f, 11.78125f, 12.375f, 12.875f, 13.375f, 13.96875f, 14.375f, 14.46875f,
         13.0f,   13.40625f, 14.0f,   14.5f,   15.0f,   15.59375f, 16.0f,   16.09375f,
         13.375f, 13.78125f, 14.375f, 14.875f, 15.375f, 15.96875f, 16.375f, 16.46875f},
        {0.45507922,  0.64057922,  0.97157922,  1.42257922,  1.90732922,  2.22332922,  2.70807922,
         3.15907922,  3.49007922,  3.67557922,  1.39437963,  1.57987963,  1.91087963,  2.36187963,
         2.84662963,  3.16262963,  3.64737963,  4.09837963,  4.42937963,  4.61487963,  2.95130693,
         3.13680693,  3.46780693,  3.91880693,  4.40355693,  4.71955693,  5.20430693,  5.65530693,
         5.98630693,  6.17180693,  5.20525069,  5.39075069,  5.72175069,  6.17275069,  6.65750069,
         6.97350069,  7.45825069,  7.90925069,  8.24025069,  8.42575069,  6.88975,     7.07525,
         7.40625,     7.85725,     8.342,       8.658,       9.14275,     9.59375,     9.92475,
         10.11025,    8.57424931,  8.75974931,  9.09074931,  9.54174931,  10.02649931, 10.34249931,
         10.82724931, 11.27824931, 11.60924931, 11.79474931, 10.82819307, 11.01369307, 11.34469307,
         11.79569307, 12.28044307, 12.59644307, 13.08119307, 13.53219307, 13.86319307, 14.04869307,
         12.38512037, 12.57062037, 12.90162037, 13.35262037, 13.83737037, 14.15337037, 14.63812037,
         15.08912037, 15.42012037, 15.60562037, 13.32442078, 13.50992078, 13.84092078, 14.29192078,
         14.77667078, 15.09267078, 15.57742078, 16.02842078, 16.35942078, 16.54492078},
        {1.0f, 2.5f, 4.0f, 7.0f, 8.5f, 10.0f, 13.0f, 14.5f, 16.0f},
        {1.0,         1.34110787,  1.80029155,  2.32944606,  2.67055394,  3.19970845,  3.65889213,
         4.0,         2.36443149,  2.70553936,  3.16472303,  3.69387755,  4.03498542,  4.56413994,
         5.02332362,  5.36443149,  4.20116618,  4.54227405,  5.00145773,  5.53061224,  5.87172012,
         6.40087464,  6.86005831,  7.20116618,  6.31778426,  6.65889213,  7.1180758,   7.64723032,
         7.98833819,  8.51749271,  8.97667638,  9.31778426,  7.68221574,  8.02332362,  8.48250729,
         9.01166181,  9.35276968,  9.8819242,   10.34110787, 10.68221574, 9.79883382,  10.13994169,
         10.59912536, 11.12827988, 11.46938776, 11.99854227, 12.45772595, 12.79883382, 11.63556851,
         11.97667638, 12.43586006, 12.96501458, 13.30612245, 13.83527697, 14.29446064, 14.63556851,
         13.0,        13.34110787, 13.80029155, 14.32944606, 14.67055394, 15.19970845, 15.65889213,
         16.0}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto image = std::make_shared<op::Parameter>(element::f32, data_shape);
        auto target_spatial_shape =
            op::Constant::create<int64_t>(element::i64, Shape{2}, s.spatial_shape);
        auto scales = op::Constant::create<float>(element::f32, Shape{2}, s.scales_data);
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

        InterpolateAttrs attrs;
        attrs.mode = InterpolateMode::cubic;
        attrs.shape_calculation_mode = s.shape_calculation_mode;
        attrs.coordinate_transformation_mode = s.transform_mode;
        attrs.nearest_mode = Nearest_mode::round_prefer_floor;
        attrs.antialias = false;
        attrs.pads_begin = {0, 0, 0, 0};
        attrs.pads_end = {0, 0, 0, 0};
        attrs.cube_coeff = -0.75;

        auto interp =
            std::make_shared<op::v4::Interpolate>(image, target_spatial_shape, scales, axes, attrs);
        auto fun = std::make_shared<Function>(OutputVector{interp}, ParameterVector{image});
        auto result = std::make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate(
            {result}, {make_host_tensor<element::Type_t::f32>(data_shape, input_data)}));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), s.out_shape);
        auto result_vector = read_vector<float>(result);
        std::size_t num_of_elems = shape_size(s.out_shape);
        for (std::size_t j = 0; j < num_of_elems; ++j)
        {
            EXPECT_NEAR(result_vector[j], expected_results[i][j], 1.2e-5);
        }
        ++i;
    }
}

TEST(op_eval, interpolate_v4_nearest)
{
    struct ShapesAndAttrs
    {
        Shape input_data_shape;
        std::vector<int64_t> spatial_shape;
        Shape out_shape;
        std::vector<float> scales_data;
        CoordinateTransformMode transform_mode;
        ShapeCalcMode shape_calculation_mode;
        Nearest_mode nearest_mode;
    };

    std::vector<ShapesAndAttrs> shapes_and_attrs = {
        // resize_downsample_scales_nearest:
        ShapesAndAttrs{Shape{1, 1, 2, 4},
                       {1, 2},
                       Shape{1, 1, 1, 2},
                       {0.6f, 0.6f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales,
                       Nearest_mode::round_prefer_floor},
        // resize_downsample_sizes_nearest:
        ShapesAndAttrs{Shape{1, 1, 2, 4},
                       {1, 2},
                       Shape{1, 1, 1, 2},
                       {0.5f, 0.5f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::sizes,
                       Nearest_mode::round_prefer_floor},
        // resize_downsample_sizes_nearest_tf_half_pixel_for_nn:
        ShapesAndAttrs{Shape{1, 1, 4, 4},
                       {3, 2},
                       Shape{1, 1, 3, 2},
                       {0.75, 0.5},
                       CoordinateTransformMode::tf_half_pixel_for_nn,
                       ShapeCalcMode::sizes,
                       Nearest_mode::round_prefer_floor},
        // resize_upsample_scales_nearest:
        ShapesAndAttrs{Shape{1, 1, 2, 2},
                       {4, 6},
                       Shape{1, 1, 4, 6},
                       {2.0f, 3.0f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales,
                       Nearest_mode::round_prefer_floor},
        // resize_upsample_sizes_nearest:
        ShapesAndAttrs{Shape{1, 1, 2, 2},
                       {7, 8},
                       Shape{1, 1, 7, 8},
                       {3.5f, 4.0f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::sizes,
                       Nearest_mode::round_prefer_floor},
        // resize_upsample_sizes_nearest_ceil_half_pixel:
        ShapesAndAttrs{Shape{1, 1, 4, 4},
                       {8, 8},
                       Shape{1, 1, 8, 8},
                       {2.0f, 2.0f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::sizes,
                       Nearest_mode::ceil},
        // resize_upsample_sizes_nearest_floor_align_corners:
        ShapesAndAttrs{Shape{1, 1, 4, 4},
                       {8, 8},
                       Shape{1, 1, 8, 8},
                       {2.0f, 2.0f},
                       CoordinateTransformMode::align_corners,
                       ShapeCalcMode::sizes,
                       Nearest_mode::floor},
        // resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric:
        ShapesAndAttrs{Shape{1, 1, 4, 4},
                       {8, 8},
                       Shape{1, 1, 8, 8},
                       {2.0f, 2.0f},
                       CoordinateTransformMode::asymmetric,
                       ShapeCalcMode::sizes,
                       Nearest_mode::round_prefer_ceil}};

    std::vector<std::vector<float>> input_data_list = {
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        {1.0f,
         2.0f,
         3.0f,
         4.0f,
         5.0f,
         6.0f,
         7.0f,
         8.0f,
         9.0f,
         10.0f,
         11.0f,
         12.0f,
         13.0f,
         14.0f,
         15.0f,
         16.0f},
        {1.0f, 2.0f, 3.0f, 4.0f},
        {1.0f, 2.0f, 3.0f, 4.0f},
        {1.0f,
         2.0f,
         3.0f,
         4.0f,
         5.0f,
         6.0f,
         7.0f,
         8.0f,
         9.0f,
         10.0f,
         11.0f,
         12.0f,
         13.0f,
         14.0f,
         15.0f,
         16.0f},
        {1.0f,
         2.0f,
         3.0f,
         4.0f,
         5.0f,
         6.0f,
         7.0f,
         8.0f,
         9.0f,
         10.0f,
         11.0f,
         12.0f,
         13.0f,
         14.0f,
         15.0f,
         16.0f},
        {1.0f,
         2.0f,
         3.0f,
         4.0f,
         5.0f,
         6.0f,
         7.0f,
         8.0f,
         9.0f,
         10.0f,
         11.0f,
         12.0f,
         13.0f,
         14.0f,
         15.0f,
         16.0f}};

    std::vector<std::vector<float>> expected_results = {
        {1.0f, 3.0f},
        {1.0f, 3.0f},
        {6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f},
        {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
         3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
         2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f,
         3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f},
        {1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  4.0f,  4.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,
         8.0f,  8.0f,  8.0f,  5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  8.0f,  8.0f,  9.0f,  10.0f,
         10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 12.0f, 9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f,
         12.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f,
         15.0f, 16.0f, 16.0f, 16.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f, 16.0f, 16.0f},
        {1.0f,  1.0f,  1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  1.0f,  1.0f,  1.0f,  2.0f, 2.0f,
         3.0f,  3.0f,  4.0f,  1.0f,  1.0f,  1.0f,  2.0f,  2.0f,  3.0f,  3.0f,  4.0f,  5.0f, 5.0f,
         5.0f,  6.0f,  6.0f,  7.0f,  7.0f,  8.0f,  5.0f,  5.0f,  5.0f,  6.0f,  6.0f,  7.0f, 7.0f,
         8.0f,  9.0f,  9.0f,  9.0f,  10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 9.0f,  9.0f,  9.0f, 10.0f,
         10.0f, 11.0f, 11.0f, 12.0f, 13.0f, 13.0f, 13.0f, 14.0f, 14.0f, 15.0f, 15.0f, 16.0f},
        {1.0,  2.0,  2.0,  3.0,  3.0,  4.0,  4.0,  4.0,  5.0,  6.0,  6.0,  7.0,  7.0,
         8.0,  8.0,  8.0,  5.0,  6.0,  6.0,  7.0,  7.0,  8.0,  8.0,  8.0,  9.0,  10.0,
         10.0, 11.0, 11.0, 12.0, 12.0, 12.0, 9.0,  10.0, 10.0, 11.0, 11.0, 12.0, 12.0,
         12.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0, 13.0, 14.0, 14.0, 15.0,
         15.0, 16.0, 16.0, 16.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto image = std::make_shared<op::Parameter>(element::f32, s.input_data_shape);
        auto target_spatial_shape =
            op::Constant::create<int64_t>(element::i64, Shape{2}, s.spatial_shape);
        auto scales = op::Constant::create<float>(element::f32, Shape{2}, s.scales_data);
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

        InterpolateAttrs attrs;
        attrs.mode = InterpolateMode::nearest;
        attrs.shape_calculation_mode = s.shape_calculation_mode;
        attrs.coordinate_transformation_mode = s.transform_mode;
        attrs.nearest_mode = s.nearest_mode;
        attrs.antialias = false;
        attrs.pads_begin = {0, 0, 0, 0};
        attrs.pads_end = {0, 0, 0, 0};
        attrs.cube_coeff = -0.75;

        auto interp =
            std::make_shared<op::v4::Interpolate>(image, target_spatial_shape, scales, axes, attrs);
        auto fun = std::make_shared<Function>(OutputVector{interp}, ParameterVector{image});
        auto result = std::make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate(
            {result},
            {make_host_tensor<element::Type_t::f32>(s.input_data_shape, input_data_list[i])}));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), s.out_shape);
        auto result_vector = read_vector<float>(result);
        std::size_t num_of_elems = shape_size(s.out_shape);
        for (std::size_t j = 0; j < num_of_elems; ++j)
        {
            EXPECT_NEAR(result_vector[j], expected_results[i][j], 0.0000002);
        }
        ++i;
    }
}

TEST(op_eval, interpolate_v4_linear_onnx)
{
    struct ShapesAndAttrs
    {
        Shape input_data_shape;
        std::vector<int64_t> spatial_shape;
        Shape out_shape;
        std::vector<float> scales_data;
        CoordinateTransformMode transform_mode;
        ShapeCalcMode shape_calculation_mode;
    };

    std::vector<ShapesAndAttrs> shapes_and_attrs = {
        // resize_downsample_scales_linear
        ShapesAndAttrs{Shape{1, 1, 2, 4},
                       {1, 2},
                       Shape{1, 1, 1, 2},
                       {0.6f, 0.6f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales},
        // resize_downsample_sizes_linear_pytorch_half_pixel
        ShapesAndAttrs{Shape{1, 1, 4, 4},
                       {3, 1},
                       Shape{1, 1, 3, 1},
                       {0.75f, 0.25f},
                       CoordinateTransformMode::pytorch_half_pixel,
                       ShapeCalcMode::sizes},
        // resize_upsample_scales_linear
        ShapesAndAttrs{Shape{1, 1, 2, 2},
                       {4, 4},
                       Shape{1, 1, 4, 4},
                       {2.0f, 2.0f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales},
        // resize_upsample_scales_linear_align_corners
        ShapesAndAttrs{Shape{1, 1, 2, 2},
                       {4, 4},
                       Shape{1, 1, 4, 4},
                       {2.0f, 2.0f},
                       CoordinateTransformMode::align_corners,
                       ShapeCalcMode::scales},
        // resize_downsample_scales_linear_align_corners:
        // (expected values from ONNX documentation are not correct!)
        ShapesAndAttrs{Shape{1, 1, 2, 4},
                       {1, 2},
                       Shape{1, 1, 1, 2},
                       {0.6f, 0.6f},
                       CoordinateTransformMode::align_corners,
                       ShapeCalcMode::scales}};

    std::vector<std::vector<float>> input_data_list = {
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        {1.0f,
         2.0f,
         3.0f,
         4.0f,
         5.0f,
         6.0f,
         7.0f,
         8.0f,
         9.0f,
         10.0f,
         11.0f,
         12.0f,
         13.0f,
         14.0f,
         15.0f,
         16.0f},
        {1.0f, 2.0f, 3.0f, 4.0f},
        {1.0f, 2.0f, 3.0f, 4.0f},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}};

    std::vector<std::vector<float>> expected_results = {{2.6666665f, 4.3333331f},
                                                        {1.6666666f, 7.0f, 12.333333f},
                                                        {1.0f,
                                                         1.25f,
                                                         1.75f,
                                                         2.0f,
                                                         1.5f,
                                                         1.75f,
                                                         2.25f,
                                                         2.5f,
                                                         2.5f,
                                                         2.75f,
                                                         3.25f,
                                                         3.5f,
                                                         3.0f,
                                                         3.25f,
                                                         3.75f,
                                                         4.0f},
                                                        {1.0f,
                                                         1.33333333f,
                                                         1.66666667f,
                                                         2.0f,
                                                         1.66666667f,
                                                         2.0f,
                                                         2.33333333f,
                                                         2.66666667f,
                                                         2.33333333f,
                                                         2.66666667f,
                                                         3.0f,
                                                         3.33333333f,
                                                         3.0f,
                                                         3.33333333f,
                                                         3.66666667f,
                                                         4.0f},
                                                        {1.0f, 4.0f}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto image = std::make_shared<op::Parameter>(element::f32, s.input_data_shape);
        auto target_spatial_shape =
            op::Constant::create<int64_t>(element::i64, Shape{2}, s.spatial_shape);
        auto scales = op::Constant::create<float>(element::f32, Shape{2}, s.scales_data);
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

        InterpolateAttrs attrs;
        attrs.mode = InterpolateMode::linear_onnx;
        attrs.shape_calculation_mode = s.shape_calculation_mode;
        attrs.coordinate_transformation_mode = s.transform_mode;
        attrs.nearest_mode = Nearest_mode::round_prefer_floor;
        attrs.antialias = false;
        attrs.pads_begin = {0, 0, 0, 0};
        attrs.pads_end = {0, 0, 0, 0};
        attrs.cube_coeff = -0.75;

        auto interp =
            std::make_shared<op::v4::Interpolate>(image, target_spatial_shape, scales, axes, attrs);
        auto fun = std::make_shared<Function>(OutputVector{interp}, ParameterVector{image});
        auto result = std::make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate(
            {result},
            {make_host_tensor<element::Type_t::f32>(s.input_data_shape, input_data_list[i])}));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), s.out_shape);
        auto result_vector = read_vector<float>(result);
        std::size_t num_of_elems = shape_size(s.out_shape);
        for (std::size_t j = 0; j < num_of_elems; ++j)
        {
            EXPECT_NEAR(result_vector[j], expected_results[i][j], 0.00001);
        }
        ++i;
    }
}

TEST(op_eval, interpolate_v4_linear_onnx5d)
{
    struct ShapesAndAttrs
    {
        Shape input_data_shape;
        std::vector<int64_t> spatial_shape;
        Shape out_shape;
        std::vector<float> scales_data;
        CoordinateTransformMode transform_mode;
        ShapeCalcMode shape_calculation_mode;
    };

    std::vector<ShapesAndAttrs> shapes_and_attrs = {
        // resize_downsample_scales_linear
        {Shape{1, 1, 3, 2, 4},
         {2, 1, 2},
         Shape{1, 1, 2, 1, 2},
         {0.8f, 0.6f, 0.6f},
         CoordinateTransformMode::half_pixel,
         ShapeCalcMode::scales},
        // resize_downsample_scales_linear_align_corners
        {Shape{1, 1, 3, 2, 4},
         {2, 1, 2},
         Shape{1, 1, 2, 1, 2},
         {0.8f, 0.6f, 0.6f},
         CoordinateTransformMode::align_corners,
         ShapeCalcMode::scales},
        // resize_upsample_scales_linear
        {Shape{1, 1, 2, 2, 2},
         {4, 4, 4},
         Shape{1, 1, 4, 4, 4},
         {2.0, 2.0, 2.0},
         CoordinateTransformMode::half_pixel,
         ShapeCalcMode::scales},
        // resize_upsample_scales_linear_align_corners
        {Shape{1, 1, 2, 2, 2},
         {4, 4, 4},
         Shape{1, 1, 4, 4, 4},
         {2.0, 2.0, 2.0},
         CoordinateTransformMode::align_corners,
         ShapeCalcMode::scales},
        // resize_downsample_sizes_linear_pytorch_half_pixel
        {Shape{1, 1, 2, 4, 4},
         {1, 3, 1},
         Shape{1, 1, 1, 3, 1},
         {0.5, 0.75, 0.25},
         CoordinateTransformMode::pytorch_half_pixel,
         ShapeCalcMode::sizes}};

    std::vector<std::vector<float>> input_data_list = {
        // resize_downsample_scales_linear
        {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
        // resize_downsample_scales_linear_align_corners
        {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
        // resize_upsample_scales_linear
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        // resize_upsample_scales_linear_align_corners
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        // resize_downsample_sizes_linear_pytorch_half_pixel
        {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
         12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f,
         23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f}};

    std::vector<std::vector<float>> expected_results = {
        // resize_downsample_scales_linear
        {3.6666665, 5.333333, 13.666666, 15.333333},
        // resize_downsample_scales_linear_align_corners
        {1.0, 4.0, 17.0, 20.0},
        // resize_upsample_scales_linear
        {1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
         2.0, 2.25, 2.75, 3.0, 2.5, 2.75, 3.25, 3.5, 3.5, 3.75, 4.25, 4.5, 4.0, 4.25, 4.75, 5.0,
         4.0, 4.25, 4.75, 5.0, 4.5, 4.75, 5.25, 5.5, 5.5, 5.75, 6.25, 6.5, 6.0, 6.25, 6.75, 7.0,
         5.0, 5.25, 5.75, 6.0, 5.5, 5.75, 6.25, 6.5, 6.5, 6.75, 7.25, 7.5, 7.0, 7.25, 7.75, 8.0},
        // resize_upsample_scales_linear_align_corners
        {1.0,       1.3333333, 1.6666667, 2.0,       1.6666666, 2.0,       2.3333335, 2.6666667,
         2.3333333, 2.6666665, 3.0,       3.3333335, 3.0,       3.3333333, 3.6666665, 4.0,
         2.3333335, 2.6666665, 3.0,       3.3333333, 3.0,       3.333333,  3.6666665, 3.9999995,
         3.6666665, 4.0,       4.3333335, 4.6666665, 4.333333,  4.6666665, 4.9999995, 5.333333,
         3.6666667, 4.0,       4.3333335, 4.6666665, 4.3333335, 4.6666665, 5.0,       5.333333,
         5.0,       5.3333335, 5.666667,  6.0,       5.666667,  5.9999995, 6.333333,  6.666667,
         5.0,       5.333333,  5.6666665, 6.0,       5.666667,  5.9999995, 6.333333,  6.666666,
         6.3333335, 6.666666,  7.0,       7.3333335, 7.0,       7.333333,  7.6666675, 8.0},
        // resize_downsample_sizes_linear_pytorch_half_pixel
        {1.6666667, 7.0, 12.333333}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto image = std::make_shared<op::Parameter>(element::f32, s.input_data_shape);
        auto target_spatial_shape =
            op::Constant::create<int64_t>(element::i64, Shape{3}, s.spatial_shape);
        auto scales = op::Constant::create<float>(element::f32, Shape{3}, s.scales_data);
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 4});

        InterpolateAttrs attrs;
        attrs.mode = InterpolateMode::linear_onnx;
        attrs.shape_calculation_mode = s.shape_calculation_mode;
        attrs.coordinate_transformation_mode = s.transform_mode;
        attrs.nearest_mode = Nearest_mode::round_prefer_floor;
        attrs.antialias = false;
        attrs.pads_begin = {0, 0, 0, 0, 0};
        attrs.pads_end = {0, 0, 0, 0, 0};
        attrs.cube_coeff = -0.75;

        auto interp =
            std::make_shared<op::v4::Interpolate>(image, target_spatial_shape, scales, axes, attrs);
        auto fun = std::make_shared<Function>(OutputVector{interp}, ParameterVector{image});
        auto result = std::make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate(
            {result},
            {make_host_tensor<element::Type_t::f32>(s.input_data_shape, input_data_list[i])}));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), s.out_shape);
        auto result_vector = read_vector<float>(result);
        std::size_t num_of_elems = shape_size(s.out_shape);
        for (std::size_t j = 0; j < num_of_elems; ++j)
        {
            EXPECT_NEAR(result_vector[j], expected_results[i][j], 0.00001);
        }
        ++i;
    }
}
