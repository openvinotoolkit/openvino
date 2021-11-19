// Copyright (C) 2021 Intel Corporation
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
    std::vector<int64_t> new_spatial_shape { 240, 450};
    ngraph::Shape scales_as_shape_elements {2, 3};
    std::vector<float> scales_as_floats {2.0f, 3.0f};
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        ngraph::OutputVector concat_1_inputs_vec(2 + 2 * (input_rank - 2));
        auto unsqueeze_before_concat_1_axes = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
        auto unsqueeze_before_concat_1 = std::make_shared<ngraph::opset8::Unsqueeze>(strided_slice_node, unsqueeze_before_concat_1_axes);
        concat_1_inputs_vec[0] = unsqueeze_before_concat_1;
        std::vector<int64_t> constants_for_concat_1(2 + 2 * (input_rank - 2), 1);
        for (size_t i = 1; i <= input_rank - 2; ++i) {
            constants_for_concat_1[2 * (i - 1) + 1] = static_cast<int64_t>(input_shape[i]);
        }
        constants_for_concat_1.back() = static_cast<int64_t>(input_shape.back());
        for (size_t i = 1; i < 2 + 2 * (input_rank - 2); ++i) {
            const auto unsqueezed_const = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{constants_for_concat_1[i]});
            const auto unsqueeze_axis = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            const auto current_unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(unsqueezed_const, unsqueeze_axis);
            concat_1_inputs_vec[i] = current_unsqueeze;
        }
        auto concat_1 = std::make_shared<ngraph::opset8::Concat>(concat_1_inputs_vec, 0);

        auto reshape_1 = std::make_shared<ngraph::opset8::Reshape>(input, concat_1);

        ngraph::OutputVector concat_2_inputs_vec(input_rank);
        auto unsqueeze_before_concat_2_axes = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
        auto unsqueeze_before_concat_2 = std::make_shared<ngraph::opset8::Unsqueeze>(strided_slice_node, unsqueeze_before_concat_2_axes);
        concat_2_inputs_vec[0] = unsqueeze_before_concat_2;
        std::vector<int64_t> constants_for_concat_2(input_rank);
        for (size_t i = 1; i <= input_rank - 2; ++i) {
            constants_for_concat_2[i] = new_spatial_shape[i];
        }
        constants_for_concat_2.back() = static_cast<int64_t>(input_shape.back());
        for (size_t i = 1; i < input_rank; ++i) {
            const auto unsqueezed_const = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{constants_for_concat_2[i]});
            const auto unsqueeze_axis = ngraph::opset8::Constant::create(ngraph::element::i64, {}, std::vector<int64_t>{0});
            const auto current_unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(unsqueezed_const, unsqueeze_axis);
            concat_2_inputs_vec[i] = current_unsqueeze;
        }
        auto concat_2 = std::make_shared<ngraph::opset8::Concat>(concat_2_inputs_vec, 0);

        ngraph::Shape mul_const_shape(2 + 2 * (input_rank - 2), static_cast<size_t>(1));
        std::vector<int64_t> mul_const_value(ngraph::shape_size(scales_as_shape_elements), static_cast<int64_t>(1));
        for (uint64_t i = 1; i <= input_rank - 2; ++i) {
            mul_const_shape[2 * i] = scales_as_shape_elements[2 * (i - 1)];
        }
        const auto mul_const = ngraph::opset8::Constant::create(ngraph::element::i64, mul_const_shape, mul_const_value);
        const auto mul = std::make_shared<ngraph::opset8::Multiply>(reshape_1, mul_const);

        auto reshape_2 = std::make_shared<ngraph::opset8::Reshape>(mul, concat_2);
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
