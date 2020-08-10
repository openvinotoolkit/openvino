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
        ShapesAndAttrs{{3, 3},
                       Shape{1, 1, 3, 3},
                       {0.8f, 0.8f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales}};

    std::vector<float> input_data = {
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    std::vector<int64_t> interp_axes = {2, 3};

    std::vector<std::vector<float>> expected_results = {{1.47119141f,
                                                         2.78125f,
                                                         4.08251953f,
                                                         6.71142578f,
                                                         8.02148438f,
                                                         9.32275391f,
                                                         11.91650391f,
                                                         13.2265625f,
                                                         14.52783203f}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto image = std::make_shared<op::Parameter>(element::f32, data_shape);
        auto target_spatial_shape =
            op::Constant::create<int64_t>(element::i64, Shape{2}, s.spatial_shape);
        auto scales = op::Constant::create<float>(element::f32, Shape{2}, s.scales_data);
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, interp_axes);

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
            {result},
            {make_host_tensor<element::Type_t::f32>(data_shape, input_data)}));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), s.out_shape);
        // ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_results[i]));
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
        ShapesAndAttrs{Shape{1, 1, 2, 4},
                       {1, 2},
                       Shape{1, 1, 1, 2},
                       {0.6f, 0.6f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales,
                       Nearest_mode::round_prefer_floor}};

    std::vector<std::vector<float>> input_data_list = {
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}};

    std::vector<int64_t> interp_axes = {2, 3};

    std::vector<std::vector<float>> expected_results = {{1.0f, 3.0f}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto image = std::make_shared<op::Parameter>(element::f32, s.input_data_shape);
        auto target_spatial_shape =
            op::Constant::create<int64_t>(element::i64, Shape{2}, s.spatial_shape);
        auto scales = op::Constant::create<float>(element::f32, Shape{2}, s.scales_data);
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, interp_axes);

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
        // ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_results[i]));
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
        ShapesAndAttrs{Shape{1, 1, 2, 4},
                       {1, 2},
                       Shape{1, 1, 1, 2},
                       {0.6f, 0.6f},
                       CoordinateTransformMode::half_pixel,
                       ShapeCalcMode::scales}};

    std::vector<std::vector<float>> input_data_list = {
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}};

    std::vector<int64_t> interp_axes = {2, 3};

    std::vector<std::vector<float>> expected_results = {{2.6666665f, 4.3333331f}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto image = std::make_shared<op::Parameter>(element::f32, s.input_data_shape);
        auto target_spatial_shape =
            op::Constant::create<int64_t>(element::i64, Shape{2}, s.spatial_shape);
        auto scales = op::Constant::create<float>(element::f32, Shape{2}, s.scales_data);
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, interp_axes);

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
        // ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_results[i]));
        ++i;
    }
}
