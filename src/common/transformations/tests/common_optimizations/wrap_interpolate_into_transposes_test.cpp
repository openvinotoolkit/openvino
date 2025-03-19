// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/wrap_interpolate_into_transposes.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DScales) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    Shape input_shape{1, 100, 120, 150};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 60});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{1, 2});

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 60});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{2, 3});

        auto first_transpose_perm = opset8::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto last_transpose_perm = opset8::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});

        auto first_transpose = std::make_shared<opset8::Transpose>(input, first_transpose_perm);
        auto interpolate =
            std::make_shared<opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<opset8::Transpose>(interpolate, last_transpose_perm);

        model_ref = std::make_shared<ov::Model>(NodeVector{last_transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DSizes) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    Shape input_shape{1, 100, 120, 150};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 60});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{1, 2});

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 60});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{2, 3});

        auto first_transpose_perm = opset8::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto last_transpose_perm = opset8::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});

        auto first_transpose = std::make_shared<opset8::Transpose>(input, first_transpose_perm);
        auto interpolate =
            std::make_shared<opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<opset8::Transpose>(interpolate, last_transpose_perm);

        model_ref = std::make_shared<ov::Model>(NodeVector{last_transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DScales) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    Shape input_shape{1, 100, 120, 150, 18};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {60, 240, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.6, 2.0, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 2, 3});

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {60, 240, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.6, 2.0, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{2, 3, 4});

        auto first_transpose_perm = opset8::Constant::create(element::i64, Shape{5}, {0, 4, 1, 2, 3});
        auto last_transpose_perm = opset8::Constant::create(element::i64, Shape{5}, {0, 2, 3, 4, 1});

        auto first_transpose = std::make_shared<opset8::Transpose>(input, first_transpose_perm);
        auto interpolate =
            std::make_shared<opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<opset8::Transpose>(interpolate, last_transpose_perm);

        model_ref = std::make_shared<ov::Model>(NodeVector{last_transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DSizes) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    Shape input_shape{1, 100, 120, 150, 18};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {60, 240, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.6, 2.0, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 2, 3});

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {60, 240, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.6, 2.0, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{2, 3, 4});

        auto first_transpose_perm = opset8::Constant::create(element::i64, Shape{5}, {0, 4, 1, 2, 3});
        auto last_transpose_perm = opset8::Constant::create(element::i64, Shape{5}, {0, 2, 3, 4, 1});

        auto first_transpose = std::make_shared<opset8::Transpose>(input, first_transpose_perm);
        auto interpolate =
            std::make_shared<opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<opset8::Transpose>(interpolate, last_transpose_perm);

        model_ref = std::make_shared<ov::Model>(NodeVector{last_transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DScalesDynamic) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    PartialShape input_shape{1, Dimension::dynamic(), 120, Dimension::dynamic()};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 60});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{1, 2});

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 60});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{2, 3});

        auto first_transpose_perm = opset8::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto last_transpose_perm = opset8::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});

        auto first_transpose = std::make_shared<opset8::Transpose>(input, first_transpose_perm);
        auto interpolate =
            std::make_shared<opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<opset8::Transpose>(interpolate, last_transpose_perm);

        model_ref = std::make_shared<ov::Model>(NodeVector{last_transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DSizesDynamic) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    PartialShape input_shape{1, Dimension::dynamic(), 120, Dimension::dynamic()};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 60});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{1, 2});

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 60});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{2, 3});

        auto first_transpose_perm = opset8::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto last_transpose_perm = opset8::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});

        auto first_transpose = std::make_shared<opset8::Transpose>(input, first_transpose_perm);
        auto interpolate =
            std::make_shared<opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<opset8::Transpose>(interpolate, last_transpose_perm);

        model_ref = std::make_shared<ov::Model>(NodeVector{last_transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DScalesDynamic) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    PartialShape input_shape{Dimension::dynamic(), 100, Dimension::dynamic(), Dimension::dynamic(), 18};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {60, 240, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.6, 2.0, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 2, 3});

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {60, 240, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.6, 2.0, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{2, 3, 4});

        auto first_transpose_perm = opset8::Constant::create(element::i64, Shape{5}, {0, 4, 1, 2, 3});
        auto last_transpose_perm = opset8::Constant::create(element::i64, Shape{5}, {0, 2, 3, 4, 1});

        auto first_transpose = std::make_shared<opset8::Transpose>(input, first_transpose_perm);
        auto interpolate =
            std::make_shared<opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<opset8::Transpose>(interpolate, last_transpose_perm);

        model_ref = std::make_shared<ov::Model>(NodeVector{last_transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DSizesDynamic) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    PartialShape input_shape{Dimension::dynamic(), 100, Dimension::dynamic(), Dimension::dynamic(), 18};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {60, 240, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.6, 2.0, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 2, 3});

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);

        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {60, 240, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.6, 2.0, 0.5});
        auto axis_node = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{2, 3, 4});

        auto first_transpose_perm = opset8::Constant::create(element::i64, Shape{5}, {0, 4, 1, 2, 3});
        auto last_transpose_perm = opset8::Constant::create(element::i64, Shape{5}, {0, 2, 3, 4, 1});

        auto first_transpose = std::make_shared<opset8::Transpose>(input, first_transpose_perm);
        auto interpolate =
            std::make_shared<opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<opset8::Transpose>(interpolate, last_transpose_perm);

        model_ref = std::make_shared<ov::Model>(NodeVector{last_transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DScalesNotApplicable) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    Shape input_shape{1, 100, 120, 150};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});

        auto range_start = opset8::Constant::create(element::i64, Shape{}, {0});
        auto range_stop = opset8::Constant::create(element::i64, Shape{}, {4});
        auto range_step = opset8::Constant::create(element::i64, Shape{}, {1});
        auto range = std::make_shared<opset8::Range>(range_start, range_stop, range_step, element::i64);

        auto indices = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{1, 3});
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});

        auto range_start = opset8::Constant::create(element::i64, Shape{}, {0});
        auto range_stop = opset8::Constant::create(element::i64, Shape{}, {4});
        auto range_step = opset8::Constant::create(element::i64, Shape{}, {1});
        auto range = std::make_shared<opset8::Range>(range_start, range_stop, range_step, element::i64);

        auto indices = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{1, 3});
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DSizesNotApplicable) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    Shape input_shape{1, 100, 120, 150};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});

        auto range_start = opset8::Constant::create(element::i64, Shape{}, {0});
        auto range_stop = opset8::Constant::create(element::i64, Shape{}, {4});
        auto range_step = opset8::Constant::create(element::i64, Shape{}, {1});
        auto range = std::make_shared<opset8::Range>(range_start, range_stop, range_step, element::i64);

        auto indices = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{1, 3});
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{2}, {50, 75});
        auto scales_node = opset8::Constant::create(element::f32, Shape{2}, {0.5, 0.5});

        auto range_start = opset8::Constant::create(element::i64, Shape{}, {0});
        auto range_stop = opset8::Constant::create(element::i64, Shape{}, {4});
        auto range_step = opset8::Constant::create(element::i64, Shape{}, {1});
        auto range = std::make_shared<opset8::Range>(range_start, range_stop, range_step, element::i64);

        auto indices = opset8::Constant::create(element::i64, {2}, std::vector<int64_t>{1, 3});
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DScalesNotApplicable) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    Shape input_shape{1, 100, 120, 150, 800};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {50, 75, 600});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.5, 0.5, 0.75});

        auto range_start = opset8::Constant::create(element::i64, Shape{}, {0});
        auto range_stop = opset8::Constant::create(element::i64, Shape{}, {5});
        auto range_step = opset8::Constant::create(element::i64, Shape{}, {1});
        auto range = std::make_shared<opset8::Range>(range_start, range_stop, range_step, element::i64);

        auto indices = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 3, 4});
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {50, 75, 600});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.5, 0.5, 0.75});

        auto range_start = opset8::Constant::create(element::i64, Shape{}, {0});
        auto range_stop = opset8::Constant::create(element::i64, Shape{}, {5});
        auto range_step = opset8::Constant::create(element::i64, Shape{}, {1});
        auto range = std::make_shared<opset8::Range>(range_start, range_stop, range_step, element::i64);

        auto indices = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 3, 4});
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DSizesNotApplicable) {
    opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    Shape input_shape{1, 100, 120, 150, 800};
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {50, 75, 600});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.5, 0.5, 0.75});

        auto range_start = opset8::Constant::create(element::i64, Shape{}, {0});
        auto range_stop = opset8::Constant::create(element::i64, Shape{}, {5});
        auto range_step = opset8::Constant::create(element::i64, Shape{}, {1});
        auto range = std::make_shared<opset8::Range>(range_start, range_stop, range_step, element::i64);

        auto indices = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 3, 4});
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
        manager.register_pass<ov::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto sizes_node = opset8::Constant::create(element::i64, Shape{3}, {50, 75, 600});
        auto scales_node = opset8::Constant::create(element::f32, Shape{3}, {0.5, 0.5, 0.75});

        auto range_start = opset8::Constant::create(element::i64, Shape{}, {0});
        auto range_stop = opset8::Constant::create(element::i64, Shape{}, {5});
        auto range_step = opset8::Constant::create(element::i64, Shape{}, {1});
        auto range = std::make_shared<opset8::Range>(range_start, range_stop, range_step, element::i64);

        auto indices = opset8::Constant::create(element::i64, {3}, std::vector<int64_t>{1, 3, 4});
        auto gather_axis_node = opset8::Constant::create(element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        model_ref = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});
    }
}
