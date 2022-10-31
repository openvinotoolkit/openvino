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
#include <transformations/common_optimizations/interpolate_sequence_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

using Attrs = ngraph::opset8::Interpolate::InterpolateAttrs;
using ShapeCalcMode = ngraph::opset8::Interpolate::ShapeCalcMode;
using InterpolateMode = ngraph::opset8::Interpolate::InterpolateMode;
using CoordinateTransformMode = ngraph::opset8::Interpolate::CoordinateTransformMode;
using NearestMode = ngraph::opset8::Interpolate::NearestMode;

TEST_F(TransformationTestsF, InterpolateSequenceFusion4D1) {
    ngraph::Shape input_shape { 1, 4, 220, 350 };
    std::vector<Attrs> attributes = {
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f},
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f}
    };
    std::vector<std::vector<int64_t>> sizes_vector = {
        {660}, {700}
    };
    std::vector<std::vector<float>> scales_vector = {
        {3.0f}, {2.0f}
    };
    std::vector<std::vector<int64_t>> axes_vector = {
        {2}, {3}
    };
    Attrs ref_attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR, false, -0.75f};
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto fst_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<ngraph::opset8::Interpolate>(fst_interpolate, snd_sizes_node, snd_scales_node, snd_axis_node, attributes[1]);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ snd_interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, std::vector<float>{3.0f, 2.0f});
        auto axes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<int64_t>{2, 3});

        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(gather_node, ngraph::element::f32);

        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto eps_node = ngraph::opset8::Constant::create(ngraph::element::f32, {}, std::vector<float>{1.0e-5f});
        auto add_node = std::make_shared<ngraph::opset8::Multiply>(mul_node, eps_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(add_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axes_node, ref_attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, InterpolateSequenceFusion4D2) {
    ngraph::Shape input_shape { 1, 4, 220, 350 };
    std::vector<Attrs> attributes = {
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f},
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f},
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f}
    };
    std::vector<std::vector<int64_t>> sizes_vector = {
        {660}, {700}, {1320}
    };
    std::vector<std::vector<float>> scales_vector = {
        {3.0f}, {2.0f}, {2.0f}
    };
    std::vector<std::vector<int64_t>> axes_vector = {
        {2}, {3}, {2}
    };
    Attrs ref_attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR, false, -0.75f};
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto fst_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<ngraph::opset8::Interpolate>(fst_interpolate, snd_sizes_node, snd_scales_node, snd_axis_node, attributes[1]);

        auto third_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[2].size()}, sizes_vector[2]);
        auto third_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[2].size()}, scales_vector[2]);
        auto third_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[2].size()}, axes_vector[2]);
        auto third_interpolate = std::make_shared<ngraph::opset8::Interpolate>(snd_interpolate, third_sizes_node, third_scales_node, third_axis_node,
                                                                               attributes[2]);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ third_interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, std::vector<float>{3.0f, 2.0f});
        auto axes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<int64_t>{2, 3});

        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(gather_node, ngraph::element::f32);

        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto eps_node = ngraph::opset8::Constant::create(ngraph::element::f32, {}, std::vector<float>{1.0e-5f});
        auto add_node = std::make_shared<ngraph::opset8::Multiply>(mul_node, eps_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(add_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto fst_interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axes_node, ref_attrs);

        auto snd_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[2].size()}, sizes_vector[2]);
        auto snd_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[2].size()}, scales_vector[2]);
        auto snd_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[2].size()}, axes_vector[2]);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(fst_interpolate, snd_sizes_node, snd_scales_node, snd_axis_node, ref_attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, InterpolateSequenceFusion4D3) {
    ngraph::Shape input_shape { 1, 4, 220, 350 };
    std::vector<Attrs> attributes = {
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f},
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f}
    };
    std::vector<std::vector<int64_t>> sizes_vector = {
        {700}, {660}
    };
    std::vector<std::vector<float>> scales_vector = {
        {2.0f}, {3.0f}
    };
    std::vector<std::vector<int64_t>> axes_vector = {
        {3}, {2}
    };
    Attrs ref_attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR, false, -0.75f};
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto fst_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<ngraph::opset8::Interpolate>(fst_interpolate, snd_sizes_node, snd_scales_node, snd_axis_node, attributes[1]);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ snd_interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{660, 700});
        auto axes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{2, 3});
        auto sizes_cast = std::make_shared<ngraph::opset8::Convert>(sizes_node, ngraph::element::f32);
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(gather_node, ngraph::element::f32);
        auto div_node = std::make_shared<ngraph::opset8::Divide>(sizes_cast, cast_shape_to_float);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, div_node, axes_node, ref_attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, InterpolateSequenceFusion5D1) {
    ngraph::Shape input_shape { 1, 5, 417, 256, 800 };
    std::vector<Attrs> attributes = {
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f},
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f},
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f}
    };
    std::vector<std::vector<int64_t>> sizes_vector = {
        {600}, {100}, {834}
    };
    std::vector<std::vector<float>> scales_vector = {
        {0.75f}, {20.0f}, {2.0f}
    };
    std::vector<std::vector<int64_t>> axes_vector = {
        {4}, {1}, {2}
    };
    Attrs ref_attrs{InterpolateMode::NEAREST, ShapeCalcMode::SCALES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR, false, -0.75f};
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto fst_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<ngraph::opset8::Interpolate>(fst_interpolate, snd_sizes_node, snd_scales_node, snd_axis_node, attributes[1]);

        auto third_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[2].size()}, sizes_vector[2]);
        auto third_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[2].size()}, scales_vector[2]);
        auto third_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[2].size()}, axes_vector[2]);
        auto third_interpolate = std::make_shared<ngraph::opset8::Interpolate>(snd_interpolate, third_sizes_node, third_scales_node, third_axis_node,
                                                                               attributes[2]);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ third_interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, std::vector<float>{20.0f, 2.0f, 0.75f});
        auto axes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{1, 2, 4});

        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(gather_node, ngraph::element::f32);

        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto eps_node = ngraph::opset8::Constant::create(ngraph::element::f32, {}, std::vector<float>{1.0e-5f});
        auto add_node = std::make_shared<ngraph::opset8::Multiply>(mul_node, eps_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(add_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axes_node, ref_attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, InterpolateSequenceFusion5D2) {
    ngraph::Shape input_shape { 1, 5, 417, 256, 800 };
    std::vector<Attrs> attributes = {
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f},
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f},
        Attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
              NearestMode::ROUND_PREFER_FLOOR, false, -0.75f}
    };
    std::vector<std::vector<int64_t>> sizes_vector = {
        {600}, {100}, {834}
    };
    std::vector<std::vector<float>> scales_vector = {
        {0.75f}, {20.0f}, {2.0f}
    };
    std::vector<std::vector<int64_t>> axes_vector = {
        {4}, {1}, {2}
    };
    Attrs ref_attrs{InterpolateMode::NEAREST, ShapeCalcMode::SIZES, std::vector<size_t>{0}, std::vector<size_t>{0}, CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR, false, -0.75f};
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto fst_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<ngraph::opset8::Interpolate>(fst_interpolate, snd_sizes_node, snd_scales_node, snd_axis_node, attributes[1]);

        auto third_sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{sizes_vector[2].size()}, sizes_vector[2]);
        auto third_scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{scales_vector[2].size()}, scales_vector[2]);
        auto third_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{axes_vector[2].size()}, axes_vector[2]);
        auto third_interpolate = std::make_shared<ngraph::opset8::Interpolate>(snd_interpolate, third_sizes_node, third_scales_node, third_axis_node,
                                                                               attributes[2]);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ third_interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{100, 834, 600});
        auto axes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 2, 4});
        auto sizes_cast = std::make_shared<ngraph::opset8::Convert>(sizes_node, ngraph::element::f32);
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(gather_node, ngraph::element::f32);
        auto div_node = std::make_shared<ngraph::opset8::Divide>(sizes_cast, cast_shape_to_float);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, div_node, axes_node, ref_attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}
