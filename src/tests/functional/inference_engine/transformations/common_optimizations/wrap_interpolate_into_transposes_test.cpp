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
#include <transformations/common_optimizations/wrap_interpolate_into_transposes.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DScales) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::Shape input_shape { 1, 100, 120, 150 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 60 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 2});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 60 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{2, 3});

        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 3, 1, 2 });
        auto last_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 2, 3, 1 });

        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<ngraph::opset8::Transpose>(interpolate, last_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ last_transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DSizes) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::Shape input_shape { 1, 100, 120, 150 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 60 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 2});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 60 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{2, 3});

        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 3, 1, 2 });
        auto last_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 2, 3, 1 });

        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<ngraph::opset8::Transpose>(interpolate, last_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ last_transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DScales) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::Shape input_shape { 1, 100, 120, 150, 18 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 60, 240, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.6, 2.0, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 2, 3});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 60, 240, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.6, 2.0, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{2, 3, 4});

        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, { 0, 4, 1, 2, 3 });
        auto last_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, { 0, 2, 3, 4, 1 });

        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<ngraph::opset8::Transpose>(interpolate, last_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ last_transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DSizes) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::Shape input_shape { 1, 100, 120, 150, 18 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 60, 240, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.6, 2.0, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 2, 3});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 60, 240, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.6, 2.0, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{2, 3, 4});

        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, { 0, 4, 1, 2, 3 });
        auto last_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, { 0, 2, 3, 4, 1 });

        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<ngraph::opset8::Transpose>(interpolate, last_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ last_transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DScalesDynamic) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::PartialShape input_shape { 1, ngraph::Dimension::dynamic(), 120, ngraph::Dimension::dynamic() };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 60 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 2});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 60 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{2, 3});

        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 3, 1, 2 });
        auto last_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 2, 3, 1 });

        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<ngraph::opset8::Transpose>(interpolate, last_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ last_transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DSizesDynamic) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::PartialShape input_shape { 1, ngraph::Dimension::dynamic(), 120, ngraph::Dimension::dynamic() };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 60 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 2});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 60 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{2, 3});

        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 3, 1, 2 });
        auto last_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{4}, { 0, 2, 3, 1 });

        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<ngraph::opset8::Transpose>(interpolate, last_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ last_transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DScalesDynamic) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::PartialShape input_shape { ngraph::Dimension::dynamic(), 100, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 18 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 60, 240, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.6, 2.0, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 2, 3});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 60, 240, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.6, 2.0, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{2, 3, 4});

        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, { 0, 4, 1, 2, 3 });
        auto last_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, { 0, 2, 3, 4, 1 });

        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<ngraph::opset8::Transpose>(interpolate, last_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ last_transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DSizesDynamic) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::PartialShape input_shape { ngraph::Dimension::dynamic(), 100, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 18 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 60, 240, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.6, 2.0, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 2, 3});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);

        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 60, 240, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.6, 2.0, 0.5 });
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{2, 3, 4});

        auto first_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, { 0, 4, 1, 2, 3 });
        auto last_transpose_perm = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{5}, { 0, 2, 3, 4, 1 });

        auto first_transpose = std::make_shared<ngraph::opset8::Transpose>(input, first_transpose_perm);
        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(first_transpose, sizes_node, scales_node, axis_node, attrs);
        auto last_transpose = std::make_shared<ngraph::opset8::Transpose>(interpolate, last_transpose_perm);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ last_transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DScalesNotApplicable) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::Shape input_shape { 1, 100, 120, 150 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });

        auto range_start = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
        auto range_stop = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 4 });
        auto range_step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 1 });
        auto range = std::make_shared<ngraph::opset8::Range>(range_start, range_stop, range_step, ngraph::element::i64);

        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 3});
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });

        auto range_start = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
        auto range_stop = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 4 });
        auto range_step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 1 });
        auto range = std::make_shared<ngraph::opset8::Range>(range_start, range_stop, range_step, ngraph::element::i64);

        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 3});
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes4DSizesNotApplicable) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::Shape input_shape { 1, 100, 120, 150 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });

        auto range_start = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
        auto range_stop = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 4 });
        auto range_step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 1 });
        auto range = std::make_shared<ngraph::opset8::Range>(range_start, range_stop, range_step, ngraph::element::i64);

        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 3});
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, { 50, 75 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{2}, { 0.5, 0.5 });

        auto range_start = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
        auto range_stop = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 4 });
        auto range_step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 1 });
        auto range = std::make_shared<ngraph::opset8::Range>(range_start, range_stop, range_step, ngraph::element::i64);

        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, {2}, std::vector<int64_t>{1, 3});
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DScalesNotApplicable) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::Shape input_shape { 1, 100, 120, 150, 800 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 50, 75, 600 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.5, 0.5, 0.75 });

        auto range_start = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
        auto range_stop = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 5 });
        auto range_step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 1 });
        auto range = std::make_shared<ngraph::opset8::Range>(range_start, range_stop, range_step, ngraph::element::i64);

        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 3, 4});
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 50, 75, 600 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.5, 0.5, 0.75 });

        auto range_start = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
        auto range_stop = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 5 });
        auto range_step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 1 });
        auto range = std::make_shared<ngraph::opset8::Range>(range_start, range_stop, range_step, ngraph::element::i64);

        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 3, 4});
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, WrapInterpolateIntoTransposes5DSizesNotApplicable) {
    ngraph::opset8::Interpolate::InterpolateAttrs attrs;

    attrs.mode = ngraph::opset8::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ngraph::opset8::Interpolate::ShapeCalcMode::SIZES;
    attrs.nearest_mode = ngraph::opset8::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = std::vector<size_t>{0};
    attrs.pads_end = std::vector<size_t>{0};
    attrs.antialias = false;
    attrs.coordinate_transformation_mode = ngraph::opset8::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.75f;

    ngraph::Shape input_shape { 1, 100, 120, 150, 800 };
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 50, 75, 600 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.5, 0.5, 0.75 });

        auto range_start = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
        auto range_stop = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 5 });
        auto range_step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 1 });
        auto range = std::make_shared<ngraph::opset8::Range>(range_start, range_stop, range_step, ngraph::element::i64);

        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 3, 4});
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::WrapInterpolateIntoTransposes>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, { 50, 75, 600 });
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, { 0.5, 0.5, 0.75 });

        auto range_start = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
        auto range_stop = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 5 });
        auto range_step = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 1 });
        auto range = std::make_shared<ngraph::opset8::Range>(range_start, range_stop, range_step, ngraph::element::i64);

        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, {3}, std::vector<int64_t>{1, 3, 4});
        auto gather_axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{0});
        auto gather_node = std::make_shared<ngraph::opset8::Gather>(range, indices, gather_axis_node);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, gather_node, attrs);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}
