// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/common_optimizations/nearest_neighbor_upsampling_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, NearestNeighborUpsamplingFusionSpatial2D1) {
    ngraph::Shape input_shape { 1, 120, 150, 32 };
    size_t input_rank = input_shape.size();
    std::vector<int64_t> new_spatial_shape { 240, 450 };
    std::vector<float> scales_as_floats { 2.0f, 3.0f };
    std::vector<int64_t> constants_for_concat_1 { 1, 120, 1, 150, 1, 32 };
    std::vector<int64_t> constants_for_concat_2 { 1, 240, 450, 32 };
    ngraph::Shape mul_const_shape {1, 1, 2, 1, 3, 1};
    std::vector<float> mul_const_value {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        ngraph::OutputVector concat_1_inputs_vec(2 + 2 * (input_rank - 2));
        concat_1_inputs_vec[0] = strided_slice_node;
        for (size_t i = 1; i < 2 + 2 * (input_rank - 2); ++i) {
            const auto unsqueezed_const = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{constants_for_concat_1[i]});
            const auto unsqueeze_axis = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            const auto current_unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(unsqueezed_const, unsqueeze_axis);
            concat_1_inputs_vec[i] = current_unsqueeze;
        }
        auto concat_1 = std::make_shared<ngraph::opset8::Concat>(concat_1_inputs_vec, 0);

        auto reshape_1 = std::make_shared<ngraph::opset8::Reshape>(input, concat_1, true);

        ngraph::OutputVector concat_2_inputs_vec(input_rank);
        concat_2_inputs_vec[0] = strided_slice_node;
        for (size_t i = 1; i < input_rank; ++i) {
            const auto unsqueezed_const = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{constants_for_concat_2[i]});
            const auto unsqueeze_axis = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            const auto current_unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(unsqueezed_const, unsqueeze_axis);
            concat_2_inputs_vec[i] = current_unsqueeze;
        }
        auto concat_2 = std::make_shared<ngraph::opset8::Concat>(concat_2_inputs_vec, 0);

        const auto mul_const = ngraph::opset8::Constant::create(ngraph::element::f32, mul_const_shape, mul_const_value);
        const auto mul = std::make_shared<ngraph::opset8::Multiply>(reshape_1, mul_const);

        auto reshape_2 = std::make_shared<ngraph::opset8::Reshape>(mul, concat_2, true);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_2 }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::NearestNeighborUpsamplingFusion>();
    }
    {
        ngraph::opset8::Interpolate::InterpolateAttrs attrs;

        attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
        attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {new_spatial_shape.size()}, new_spatial_shape);
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {scales_as_floats.size()}, scales_as_floats);
        auto axes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 2});
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axes_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, NearestNeighborUpsamplingFusionSpatial3D1) {
    ngraph::Shape input_shape { 1, 130, 120, 85, 3 };
    size_t input_rank = input_shape.size();
    std::vector<int64_t> new_spatial_shape { 260, 360, 340 };
    std::vector<float> scales_as_floats { 2.0f, 3.0, 4.0f };
    std::vector<int64_t> constants_for_concat_1 { 1, 130, 1, 120, 1, 85, 1, 3 };
    std::vector<int64_t> constants_for_concat_2 { 1, 260, 360, 340, 3 };
    ngraph::Shape mul_const_shape {1, 1, 2, 1, 3, 1, 4, 1};
    std::vector<float> mul_const_value(24, 1.0f);
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        ngraph::OutputVector concat_1_inputs_vec(2 + 2 * (input_rank - 2));
        concat_1_inputs_vec[0] = strided_slice_node;
        for (size_t i = 1; i < 2 + 2 * (input_rank - 2); ++i) {
            const auto unsqueezed_const = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{constants_for_concat_1[i]});
            const auto unsqueeze_axis = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            const auto current_unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(unsqueezed_const, unsqueeze_axis);
            concat_1_inputs_vec[i] = current_unsqueeze;
        }
        auto concat_1 = std::make_shared<ngraph::opset8::Concat>(concat_1_inputs_vec, 0);

        auto reshape_1 = std::make_shared<ngraph::opset8::Reshape>(input, concat_1, true);

        ngraph::OutputVector concat_2_inputs_vec(input_rank);
        concat_2_inputs_vec[0] = strided_slice_node;
        for (size_t i = 1; i < input_rank; ++i) {
            const auto unsqueezed_const = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{constants_for_concat_2[i]});
            const auto unsqueeze_axis = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            const auto current_unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(unsqueezed_const, unsqueeze_axis);
            concat_2_inputs_vec[i] = current_unsqueeze;
        }
        auto concat_2 = std::make_shared<ngraph::opset8::Concat>(concat_2_inputs_vec, 0);

        const auto mul_const = ngraph::opset8::Constant::create(ngraph::element::f32, mul_const_shape, mul_const_value);
        const auto mul = std::make_shared<ngraph::opset8::Multiply>(reshape_1, mul_const);

        auto reshape_2 = std::make_shared<ngraph::opset8::Reshape>(mul, concat_2, true);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_2 }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::NearestNeighborUpsamplingFusion>();
    }
    {
        ngraph::opset8::Interpolate::InterpolateAttrs attrs;

        attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
        attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {new_spatial_shape.size()}, new_spatial_shape);
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {scales_as_floats.size()}, scales_as_floats);
        auto axes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 2, 3});
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axes_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}
