// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/interpolate_sequence_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

using Attrs = opset8::Interpolate::InterpolateAttrs;
using ShapeCalcMode = opset8::Interpolate::ShapeCalcMode;
using InterpolateMode = opset8::Interpolate::InterpolateMode;
using CoordinateTransformMode = opset8::Interpolate::CoordinateTransformMode;
using NearestMode = opset8::Interpolate::NearestMode;

TEST_F(TransformationTestsF, InterpolateSequenceFusion4D1) {
    Shape input_shape{1, 4, 220, 350};
    std::vector<Attrs> attributes = {Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SCALES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f},
                                     Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SCALES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f}};
    std::vector<std::vector<int64_t>> sizes_vector = {{660}, {700}};
    std::vector<std::vector<float>> scales_vector = {{3.0f}, {2.0f}};
    std::vector<std::vector<int64_t>> axes_vector = {{2}, {3}};
    Attrs ref_attrs{InterpolateMode::NEAREST,
                    ShapeCalcMode::SCALES,
                    std::vector<size_t>{0},
                    std::vector<size_t>{0},
                    CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR,
                    false,
                    -0.75f};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto fst_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate =
            std::make_shared<opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<opset8::Interpolate>(fst_interpolate,
                                                                     snd_sizes_node,
                                                                     snd_scales_node,
                                                                     snd_axis_node,
                                                                     attributes[1]);

        model = std::make_shared<ov::Model>(NodeVector{snd_interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, std::vector<float>{3.0f, 2.0f});
        auto axes_node = opset8::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{2, 3});

        auto shape_node = std::make_shared<opset8::ShapeOf>(input);
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<opset8::Convert>(gather_node, element::f32);

        auto mul_node = std::make_shared<opset8::Multiply>(cast_shape_to_float, scales_node);
        auto eps_node = opset8::Constant::create(element::f32, {}, std::vector<float>{1.0e-5f});
        auto add_node = std::make_shared<opset8::Multiply>(mul_node, eps_node);
        auto floor_node = std::make_shared<opset8::Floor>(add_node);
        auto cast_mul_result_to_int = std::make_shared<opset8::Convert>(floor_node, element::i64);

        auto interpolate =
            std::make_shared<opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axes_node, ref_attrs);
        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, InterpolateSequenceFusion4D2) {
    Shape input_shape{1, 4, 220, 350};
    std::vector<Attrs> attributes = {Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SCALES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f},
                                     Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SCALES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f},
                                     Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SCALES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f}};
    std::vector<std::vector<int64_t>> sizes_vector = {{660}, {700}, {1320}};
    std::vector<std::vector<float>> scales_vector = {{3.0f}, {2.0f}, {2.0f}};
    std::vector<std::vector<int64_t>> axes_vector = {{2}, {3}, {2}};
    Attrs ref_attrs{InterpolateMode::NEAREST,
                    ShapeCalcMode::SCALES,
                    std::vector<size_t>{0},
                    std::vector<size_t>{0},
                    CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR,
                    false,
                    -0.75f};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto fst_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate =
            std::make_shared<opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<opset8::Interpolate>(fst_interpolate,
                                                                     snd_sizes_node,
                                                                     snd_scales_node,
                                                                     snd_axis_node,
                                                                     attributes[1]);

        auto third_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[2].size()}, sizes_vector[2]);
        auto third_scales_node =
            opset8::Constant::create(element::f32, Shape{scales_vector[2].size()}, scales_vector[2]);
        auto third_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[2].size()}, axes_vector[2]);
        auto third_interpolate = std::make_shared<opset8::Interpolate>(snd_interpolate,
                                                                       third_sizes_node,
                                                                       third_scales_node,
                                                                       third_axis_node,
                                                                       attributes[2]);

        model = std::make_shared<ov::Model>(NodeVector{third_interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, std::vector<float>{3.0f, 2.0f});
        auto axes_node = opset8::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{2, 3});

        auto shape_node = std::make_shared<opset8::ShapeOf>(input);
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<opset8::Convert>(gather_node, element::f32);

        auto mul_node = std::make_shared<opset8::Multiply>(cast_shape_to_float, scales_node);
        auto eps_node = opset8::Constant::create(element::f32, {}, std::vector<float>{1.0e-5f});
        auto add_node = std::make_shared<opset8::Multiply>(mul_node, eps_node);
        auto floor_node = std::make_shared<opset8::Floor>(add_node);
        auto cast_mul_result_to_int = std::make_shared<opset8::Convert>(floor_node, element::i64);

        auto fst_interpolate =
            std::make_shared<opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axes_node, ref_attrs);

        auto snd_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[2].size()}, sizes_vector[2]);
        auto snd_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[2].size()}, scales_vector[2]);
        auto snd_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[2].size()}, axes_vector[2]);

        auto interpolate = std::make_shared<opset8::Interpolate>(fst_interpolate,
                                                                 snd_sizes_node,
                                                                 snd_scales_node,
                                                                 snd_axis_node,
                                                                 ref_attrs);
        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, InterpolateSequenceFusion4D3) {
    Shape input_shape{1, 4, 220, 350};
    std::vector<Attrs> attributes = {Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SIZES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f},
                                     Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SIZES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f}};
    std::vector<std::vector<int64_t>> sizes_vector = {{700}, {660}};
    std::vector<std::vector<float>> scales_vector = {{2.0f}, {3.0f}};
    std::vector<std::vector<int64_t>> axes_vector = {{3}, {2}};
    Attrs ref_attrs{InterpolateMode::NEAREST,
                    ShapeCalcMode::SIZES,
                    std::vector<size_t>{0},
                    std::vector<size_t>{0},
                    CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR,
                    false,
                    -0.75f};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto fst_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate =
            std::make_shared<opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<opset8::Interpolate>(fst_interpolate,
                                                                     snd_sizes_node,
                                                                     snd_scales_node,
                                                                     snd_axis_node,
                                                                     attributes[1]);

        model = std::make_shared<ov::Model>(NodeVector{snd_interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{660, 700});
        auto axes_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{2, 3});
        auto sizes_cast = std::make_shared<opset8::Convert>(sizes_node, element::f32);
        auto shape_node = std::make_shared<opset8::ShapeOf>(input);

        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<opset8::Convert>(gather_node, element::f32);
        auto div_node = std::make_shared<opset8::Divide>(sizes_cast, cast_shape_to_float);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, div_node, axes_node, ref_attrs);
        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, InterpolateSequenceFusion5D1) {
    Shape input_shape{1, 5, 417, 256, 800};
    std::vector<Attrs> attributes = {Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SCALES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f},
                                     Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SCALES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f},
                                     Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SCALES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f}};
    std::vector<std::vector<int64_t>> sizes_vector = {{600}, {100}, {834}};
    std::vector<std::vector<float>> scales_vector = {{0.75f}, {20.0f}, {2.0f}};
    std::vector<std::vector<int64_t>> axes_vector = {{4}, {1}, {2}};
    Attrs ref_attrs{InterpolateMode::NEAREST,
                    ShapeCalcMode::SCALES,
                    std::vector<size_t>{0},
                    std::vector<size_t>{0},
                    CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR,
                    false,
                    -0.75f};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto fst_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate =
            std::make_shared<opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<opset8::Interpolate>(fst_interpolate,
                                                                     snd_sizes_node,
                                                                     snd_scales_node,
                                                                     snd_axis_node,
                                                                     attributes[1]);

        auto third_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[2].size()}, sizes_vector[2]);
        auto third_scales_node =
            opset8::Constant::create(element::f32, Shape{scales_vector[2].size()}, scales_vector[2]);
        auto third_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[2].size()}, axes_vector[2]);
        auto third_interpolate = std::make_shared<opset8::Interpolate>(snd_interpolate,
                                                                       third_sizes_node,
                                                                       third_scales_node,
                                                                       third_axis_node,
                                                                       attributes[2]);

        model = std::make_shared<ov::Model>(NodeVector{third_interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, std::vector<float>{20.0f, 2.0f, 0.75f});
        auto axes_node = opset8::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 2, 4});

        auto shape_node = std::make_shared<opset8::ShapeOf>(input);
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<opset8::Convert>(gather_node, element::f32);

        auto mul_node = std::make_shared<opset8::Multiply>(cast_shape_to_float, scales_node);
        auto eps_node = opset8::Constant::create(element::f32, {}, std::vector<float>{1.0e-5f});
        auto add_node = std::make_shared<opset8::Multiply>(mul_node, eps_node);
        auto floor_node = std::make_shared<opset8::Floor>(add_node);
        auto cast_mul_result_to_int = std::make_shared<opset8::Convert>(floor_node, element::i64);

        auto interpolate =
            std::make_shared<opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axes_node, ref_attrs);
        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, InterpolateSequenceFusion5D2) {
    Shape input_shape{1, 5, 417, 256, 800};
    std::vector<Attrs> attributes = {Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SIZES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f},
                                     Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SIZES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f},
                                     Attrs{InterpolateMode::NEAREST,
                                           ShapeCalcMode::SIZES,
                                           std::vector<size_t>{0},
                                           std::vector<size_t>{0},
                                           CoordinateTransformMode::HALF_PIXEL,
                                           NearestMode::ROUND_PREFER_FLOOR,
                                           false,
                                           -0.75f}};
    std::vector<std::vector<int64_t>> sizes_vector = {{600}, {100}, {834}};
    std::vector<std::vector<float>> scales_vector = {{0.75f}, {20.0f}, {2.0f}};
    std::vector<std::vector<int64_t>> axes_vector = {{4}, {1}, {2}};
    Attrs ref_attrs{InterpolateMode::NEAREST,
                    ShapeCalcMode::SIZES,
                    std::vector<size_t>{0},
                    std::vector<size_t>{0},
                    CoordinateTransformMode::HALF_PIXEL,
                    NearestMode::ROUND_PREFER_FLOOR,
                    false,
                    -0.75f};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto fst_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[0].size()}, sizes_vector[0]);
        auto fst_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[0].size()}, scales_vector[0]);
        auto fst_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[0].size()}, axes_vector[0]);
        auto fst_interpolate =
            std::make_shared<opset8::Interpolate>(input, fst_sizes_node, fst_scales_node, fst_axis_node, attributes[0]);

        auto snd_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[1].size()}, sizes_vector[1]);
        auto snd_scales_node = opset8::Constant::create(element::f32, Shape{scales_vector[1].size()}, scales_vector[1]);
        auto snd_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[1].size()}, axes_vector[1]);
        auto snd_interpolate = std::make_shared<opset8::Interpolate>(fst_interpolate,
                                                                     snd_sizes_node,
                                                                     snd_scales_node,
                                                                     snd_axis_node,
                                                                     attributes[1]);

        auto third_sizes_node = opset8::Constant::create(element::i64, Shape{sizes_vector[2].size()}, sizes_vector[2]);
        auto third_scales_node =
            opset8::Constant::create(element::f32, Shape{scales_vector[2].size()}, scales_vector[2]);
        auto third_axis_node = opset8::Constant::create(element::i64, Shape{axes_vector[2].size()}, axes_vector[2]);
        auto third_interpolate = std::make_shared<opset8::Interpolate>(snd_interpolate,
                                                                       third_sizes_node,
                                                                       third_scales_node,
                                                                       third_axis_node,
                                                                       attributes[2]);

        model = std::make_shared<ov::Model>(NodeVector{third_interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::InterpolateSequenceFusion>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{100, 834, 600});
        auto axes_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 2, 4});
        auto sizes_cast = std::make_shared<opset8::Convert>(sizes_node, element::f32);
        auto shape_node = std::make_shared<opset8::ShapeOf>(input);

        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(shape_node, axes_node, gather_axis_node);
        auto cast_shape_to_float = std::make_shared<opset8::Convert>(gather_node, element::f32);
        auto div_node = std::make_shared<opset8::Divide>(sizes_cast, cast_shape_to_float);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, div_node, axes_node, ref_attrs);
        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}
