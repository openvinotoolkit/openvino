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
#include <transformations/common_optimizations/split_concat_pair_to_interpolate_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial2D1) {
    ngraph::Shape input_shape { 1, 100, 120, 150 };
    int64_t axis = 3;
    size_t num_splits = input_shape[axis];
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>(false);
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial2D2) {
    ngraph::Shape input_shape { 1, 100, 120, 150 };
    int64_t axis = 2;
    size_t num_splits = input_shape[axis];
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>(false);
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial3D1) {
    ngraph::Shape input_shape { 1, 3, 100, 120, 150 };
    int64_t axis = 4;
    size_t num_splits = input_shape[axis];
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>(false);
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial3D2) {
    ngraph::Shape input_shape { 1, 3, 100, 120, 150 };
    int64_t axis = 3;
    size_t num_splits = input_shape[axis];
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>(false);
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionTwoSplitsOneConcat) {
    size_t num_splits = 2;
    int64_t axis = 4;
    {
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 13, 13, 3, 2 });
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 13, 13, 3, 2 });

        auto split1_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split1 = std::make_shared<ngraph::opset8::Split>(input1, split1_axis, num_splits);

        auto split2_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split2 = std::make_shared<ngraph::opset8::Split>(input2, split2_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec{split1->output(0), split1->output(1), split2->output(0), split2->output(1)};

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input1, input2 });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 13, 13, 3, 2 });
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 13, 13, 3, 2 });

        auto split1_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split1 = std::make_shared<ngraph::opset8::Split>(input1, split1_axis, num_splits);

        auto split2_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split2 = std::make_shared<ngraph::opset8::Split>(input2, split2_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec{split1->output(0), split1->output(1), split2->output(0), split2->output(1)};

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input1, input2 });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial2D1WithConstantFolding) {
    ngraph::Shape input_shape { 1, 100, 120, 150 };
    int64_t axis = 3;
    size_t num_splits = input_shape[axis];
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    int64_t target_size = static_cast<int64_t>(input_shape[axis]) * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>();
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{target_size});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial2D2WithConstantFolding) {
    ngraph::Shape input_shape { 1, 100, 120, 150 };
    int64_t axis = 2;
    size_t num_splits = input_shape[axis];
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    int64_t target_size = static_cast<int64_t>(input_shape[axis]) * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>();
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{target_size});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial3D1WithConstantFolding) {
    ngraph::Shape input_shape { 1, 3, 100, 120, 150 };
    int64_t axis = 4;
    size_t num_splits = input_shape[axis];
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    int64_t target_size = static_cast<int64_t>(input_shape[axis]) * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>();
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{target_size});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial3D2WithConstantFolding) {
    ngraph::Shape input_shape { 1, 3, 100, 120, 150 };
    int64_t axis = 3;
    size_t num_splits = input_shape[axis];
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    int64_t target_size = static_cast<int64_t>(input_shape[axis]) * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>();
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sizes_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{target_size});

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, sizes_node, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial2D1Dynamic) {
    ngraph::PartialShape input_shape { ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 150 };
    int64_t axis = 3;
    size_t num_splits = input_shape[axis].get_length();
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>(false);
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial2D2Dynamic) {
    ngraph::PartialShape input_shape { ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 120, ngraph::Dimension::dynamic() };
    int64_t axis = 2;
    size_t num_splits = input_shape[axis].get_length();
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>(false);
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial3D1Dynamic) {
    ngraph::PartialShape input_shape { ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(),
                                       ngraph::Dimension::dynamic(), 150 };
    int64_t axis = 4;
    size_t num_splits = input_shape[axis].get_length();
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>(false);
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSpatial3D2Dynamic) {
    ngraph::PartialShape input_shape { ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(),
                                       120, ngraph::Dimension::dynamic() };
    int64_t axis = 3;
    size_t num_splits = input_shape[axis].get_length();
    size_t scale_factor = 2;
    size_t num_of_concat_inputs = num_splits * scale_factor;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
        auto split_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, { axis });
        auto split = std::make_shared<ngraph::opset8::Split>(input, split_axis, num_splits);

        ngraph::OutputVector concat_inputs_vec(num_of_concat_inputs);
        for (size_t split_output_port = 0; split_output_port < num_splits; ++split_output_port) {
            for (size_t j = 0; j < scale_factor; ++j) {
                concat_inputs_vec[split_output_port * scale_factor + j] = split->output(split_output_port);
            }
        }

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ concat }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>(false);
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
        auto scales_node = ngraph::opset8::Constant::create(ngraph::element::f32, {1}, std::vector<float>{static_cast<float>(scale_factor)});
        auto axis_node = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto shape_node = std::make_shared<ngraph::opset8::ShapeOf>(input);

        auto sslice_begin = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis});
        auto sslice_end = ngraph::opset8::Constant::create(ngraph::element::i64, {1}, std::vector<int64_t>{axis + 1});
        std::vector<int64_t> begin_mask = {0};
        std::vector<int64_t> end_mask = {0};
        auto strided_slice_node = std::make_shared<ngraph::opset8::StridedSlice>(shape_node, sslice_begin, sslice_end, begin_mask, end_mask);

        auto cast_shape_to_float = std::make_shared<ngraph::opset8::Convert>(strided_slice_node, ngraph::element::f32);
        auto mul_node = std::make_shared<ngraph::opset8::Multiply>(cast_shape_to_float, scales_node);
        auto floor_node = std::make_shared<ngraph::opset8::Floor>(mul_node);
        auto cast_mul_result_to_int = std::make_shared<ngraph::opset8::Convert>(floor_node, ngraph::element::i64);

        auto interpolate = std::make_shared<ngraph::opset8::Interpolate>(input, cast_mul_result_to_int, scales_node, axis_node, attrs);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ interpolate }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, SplitConcatPairToInterpolateFusionSplitWithEmptyPorts) {
    // it covers a case with Split node of which some outputs ports are disconnected
    // in this case the transformation is not applied
    size_t num_splits = 3;
    int64_t axis = 1;
    {
        auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 96, 96});

        auto split1_axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});
        auto split1 = std::make_shared<ngraph::opset8::Split>(input1, split1_axis, num_splits);

        auto concat_const1 = ngraph::opset8::Constant::create(ngraph::element::f32,
                                                              ngraph::Shape{1, 1, 96, 96},
                                                              std::vector<float>(96 * 96, 0));
        auto concat_const2 = ngraph::opset8::Constant::create(ngraph::element::f32,
                                                              ngraph::Shape{1, 1, 96, 96},
                                                              std::vector<float>(96 * 96, 0));

        ngraph::OutputVector concat_inputs_vec{split1->output(0), concat_const1, concat_const2};

        auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs_vec, axis);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1});
        manager.register_pass<ngraph::pass::SplitConcatPairToInterpolateFusion>();
    }
}
