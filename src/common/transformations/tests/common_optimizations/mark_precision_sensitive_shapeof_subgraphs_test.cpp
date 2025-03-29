// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, MarkEntireShapeSubgraphs_trivial_case) {
    // check that marking does not leak in trivial case
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto new_shape = std::make_shared<opset10::ShapeOf>(input_2);
        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto new_shape = std::make_shared<opset10::ShapeOf>(input_2);
        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, MarkEntireShapeSubgraphs_whole_shape_subgraph_is_marked_1) {
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof = std::make_shared<opset10::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset10::Convert>(shapeof, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset10::Convert>(div, element::i64);

        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset10::Convert>(shapeof_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset10::Convert>(div, element::i64);

        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);
        disable_fp16_compression(convert_to_float);
        disable_fp16_compression(const_denominator);
        disable_fp16_compression(div);
        disable_fp16_compression(new_shape);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, MarkEntireShapeSubgraphs_whole_shape_subgraph_is_marked_2) {
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto indices = opset10::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset10::Convert>(div, element::i64);

        auto const_ends = opset10::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset10::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset10::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset10::Result>(slice);
        model = std::make_shared<Model>(NodeVector{result}, ParameterVector{input_1});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto indices = opset10::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset10::Convert>(div, element::i64);

        auto const_ends = opset10::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset10::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset10::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset10::Result>(slice);

        disable_fp16_compression(gather_1);
        disable_fp16_compression(convert_to_float);
        disable_fp16_compression(const_denominator);
        disable_fp16_compression(div);
        disable_fp16_compression(new_dim_size);
        disable_fp16_compression(const_ends);
        disable_fp16_compression(concat_with_ends);
        model_ref = std::make_shared<Model>(NodeVector{result}, ParameterVector{input_1});
    }
}

TEST_F(TransformationTestsF, MarkEntireShapeSubgraphs_whole_shape_subgraph_is_marked_3) {
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto shapeof_2 = std::make_shared<opset10::ShapeOf>(input_2);

        auto const_1 = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto axis_const = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_2, const_1, axis_const);
        auto convert_1 = std::make_shared<opset10::Convert>(shapeof_1, element::f32);

        auto convert_2 = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_2 = opset10::Constant::create(element::f32, Shape{2}, {512, 512});
        auto div_1 = std::make_shared<opset10::Divide>(const_2, convert_2);
        auto const_3 = opset10::Constant::create(element::f32, Shape{2}, {1, 1});
        auto concat = std::make_shared<opset10::Concat>(OutputVector{const_3, div_1}, 0);  // scales

        auto mul_1 = std::make_shared<opset10::Multiply>(convert_1, concat);
        auto convert_3 = std::make_shared<opset10::Convert>(mul_1, element::i64);  // sizes

        opset10::Interpolate::InterpolateAttrs attrs;
        attrs.mode = opset10::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SIZES;
        attrs.nearest_mode = opset10::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset10::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto interpolate = std::make_shared<opset10::Interpolate>(input_1, convert_3, concat, attrs);

        auto const_4 = opset10::Constant::create(element::f32, Shape{}, {0.1f});
        auto add_1 = std::make_shared<opset10::Add>(input_1, const_4);
        auto result_1 = std::make_shared<opset10::Result>(add_1);
        auto result_2 = std::make_shared<opset10::Result>(interpolate);
        model = std::make_shared<Model>(NodeVector{result_1, result_2}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto shapeof_2 = std::make_shared<opset10::ShapeOf>(input_2);

        auto const_1 = opset10::Constant::create(element::i64, Shape{2}, {2, 3});
        auto axis_const = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_2, const_1, axis_const);
        auto convert_1 = std::make_shared<opset10::Convert>(shapeof_1, element::f32);

        auto convert_2 = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_2 = opset10::Constant::create(element::f32, Shape{2}, {512, 512});
        auto div_1 = std::make_shared<opset10::Divide>(const_2, convert_2);
        auto const_3 = opset10::Constant::create(element::f32, Shape{2}, {1, 1});
        auto concat = std::make_shared<opset10::Concat>(OutputVector{const_3, div_1}, 0);  // scales

        auto mul_1 = std::make_shared<opset10::Multiply>(convert_1, concat);
        auto convert_3 = std::make_shared<opset10::Convert>(mul_1, element::i64);  // sizes

        opset10::Interpolate::InterpolateAttrs attrs;
        attrs.mode = opset10::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SIZES;
        attrs.nearest_mode = opset10::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset10::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto interpolate = std::make_shared<opset10::Interpolate>(input_1, convert_3, concat, attrs);

        auto const_4 = opset10::Constant::create(element::f32, Shape{}, {0.1f});
        auto add_1 = std::make_shared<opset10::Add>(input_1, const_4);
        auto result_1 = std::make_shared<opset10::Result>(add_1);
        auto result_2 = std::make_shared<opset10::Result>(interpolate);

        disable_fp16_compression(gather_1);
        disable_fp16_compression(convert_1);
        disable_fp16_compression(convert_2);
        disable_fp16_compression(const_2);
        disable_fp16_compression(div_1);
        disable_fp16_compression(const_3);
        disable_fp16_compression(concat);
        disable_fp16_compression(mul_1);
        disable_fp16_compression(convert_3);

        model_ref = std::make_shared<Model>(NodeVector{result_1, result_2}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, MarkConstantsInShapeSubgraphs_only_consts_marked_1) {
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof = std::make_shared<opset10::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset10::Convert>(shapeof, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset10::Convert>(div, element::i64);
        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);

        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveConstants>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset10::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset10::Convert>(shapeof_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset10::Convert>(div, element::i64);
        auto reshape = std::make_shared<opset10::Reshape>(input_1, new_shape, false);

        disable_fp16_compression(const_denominator);

        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }
}

TEST_F(TransformationTestsF, MarkConstantsInShapeSubgraphs_only_consts_marked_2) {
    // check that only constants are marked
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto indices = opset10::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset10::Convert>(div, element::i64);

        auto const_ends = opset10::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset10::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset10::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset10::Result>(slice);
        model = std::make_shared<Model>(NodeVector{result}, ParameterVector{input_1});

        pass::Manager manager;
        manager.register_pass<pass::MarkPrecisionSensitiveConstants>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset10::ShapeOf>(input_1);
        auto indices = opset10::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset10::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset10::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset10::Convert>(gather_1, element::f32);
        auto const_denominator = opset10::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset10::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset10::Convert>(div, element::i64);

        auto const_ends = opset10::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset10::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset10::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset10::Result>(slice);

        disable_fp16_compression(const_denominator);
        disable_fp16_compression(const_ends);
        model_ref = std::make_shared<Model>(NodeVector{result}, ParameterVector{input_1});
    }
}
