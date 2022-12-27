// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "common_test_utils/graph_comparator.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace testing;
using namespace ov;

TEST(TransformationTests, MarkEntireShapeSubgraphs_trivial_case) {
    // check that marking does not leak in trivial case
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto new_shape = std::make_shared<opset8::ShapeOf>(input_2);
        auto reshape = std::make_shared<opset8::Reshape>(input_1, new_shape, false);
        model = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<ov::pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto new_shape = std::make_shared<opset8::ShapeOf>(input_2);
        auto reshape = std::make_shared<opset8::Reshape>(input_1, new_shape, false);
        model_ref = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::RUNTIME_KEYS);
    const auto res = fc.compare(model_ref, model);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, MarkEntireShapeSubgraphs_whole_shape_subgraph_is_marked_1) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof = std::make_shared<opset8::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset8::Convert>(shapeof, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset8::Convert>(div, element::i64);

        auto reshape = std::make_shared<opset8::Reshape>(input_1, new_shape, false);
        model = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<ov::pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset8::Convert>(shapeof_1, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset8::Convert>(div, element::i64);

        auto reshape = std::make_shared<opset8::Reshape>(input_1, new_shape, false);
        ov::disable_fp16_compression(convert_to_float);
        ov::disable_fp16_compression(const_denominator);
        ov::disable_fp16_compression(div);
        ov::disable_fp16_compression(new_shape);
        model_ref = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::RUNTIME_KEYS);
    const auto res = fc.compare(model_ref, model);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, MarkEntireShapeSubgraphs_whole_shape_subgraph_is_marked_2) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_1);
        auto indices = opset8::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset8::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset8::Convert>(gather_1, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset8::Convert>(div, element::i64);

        auto const_ends = opset8::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset8::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset8::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset8::Result>(slice);
        model = std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{input_1});

        pass::Manager manager;
        manager.register_pass<ov::pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_1);
        auto indices = opset8::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset8::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset8::Convert>(gather_1, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset8::Convert>(div, element::i64);

        auto const_ends = opset8::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset8::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset8::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset8::Result>(slice);

        ov::disable_fp16_compression(gather_1);
        ov::disable_fp16_compression(convert_to_float);
        ov::disable_fp16_compression(const_denominator);
        ov::disable_fp16_compression(div);
        ov::disable_fp16_compression(new_dim_size);
        ov::disable_fp16_compression(const_ends);
        ov::disable_fp16_compression(concat_with_ends);
        model_ref = std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{input_1});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model_ref, model);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, MarkEntireShapeSubgraphs_whole_shape_subgraph_is_marked_3) {
    std::shared_ptr<ov::Model> model(nullptr);
    std::shared_ptr<ov::Model> model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_1);
        auto shapeof_2 = std::make_shared<opset8::ShapeOf>(input_2);

        auto const_1 = opset8::Constant::create(element::i64, Shape{2}, {2, 3});
        auto axis_const = opset8::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shapeof_2, const_1, axis_const);
        auto convert_1 = std::make_shared<opset8::Convert>(shapeof_1, element::f32);

        auto convert_2 = std::make_shared<opset8::Convert>(gather_1, element::f32);
        auto const_2 = opset8::Constant::create(element::f32, Shape{2}, {512, 512});
        auto div_1 = std::make_shared<opset8::Divide>(const_2, convert_2);
        auto const_3 = opset8::Constant::create(element::f32, Shape{2}, {1, 1});
        auto concat = std::make_shared<opset8::Concat>(OutputVector{const_3, div_1}, 0);  // scales

        auto mul_1 = std::make_shared<opset8::Multiply>(convert_1, concat);
        auto convert_3 = std::make_shared<opset8::Convert>(mul_1, element::i64);  // sizes

        opset8::Interpolate::InterpolateAttrs attrs;
        attrs.mode = opset8::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
        attrs.nearest_mode = opset8::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto interpolate = std::make_shared<opset8::Interpolate>(input_1, convert_3, concat, attrs);

        auto const_4 = opset8::Constant::create(element::f32, Shape{}, {0.1f});
        auto add_1 = std::make_shared<opset5::Add>(input_1, const_4);
        auto result_1 = std::make_shared<opset5::Result>(add_1);
        auto result_2 = std::make_shared<opset5::Result>(interpolate);
        model = std::make_shared<ov::Model>(NodeVector{result_1, result_2}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<ov::pass::MarkPrecisionSensitiveShapeOfSubgraphs>();
        manager.run_passes(model);
    }

    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_1);
        auto shapeof_2 = std::make_shared<opset8::ShapeOf>(input_2);

        auto const_1 = opset8::Constant::create(element::i64, Shape{2}, {2, 3});
        auto axis_const = opset8::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shapeof_2, const_1, axis_const);
        auto convert_1 = std::make_shared<opset8::Convert>(shapeof_1, element::f32);

        auto convert_2 = std::make_shared<opset8::Convert>(gather_1, element::f32);
        auto const_2 = opset8::Constant::create(element::f32, Shape{2}, {512, 512});
        auto div_1 = std::make_shared<opset8::Divide>(const_2, convert_2);
        auto const_3 = opset8::Constant::create(element::f32, Shape{2}, {1, 1});
        auto concat = std::make_shared<opset8::Concat>(OutputVector{const_3, div_1}, 0);  // scales

        auto mul_1 = std::make_shared<opset8::Multiply>(convert_1, concat);
        auto convert_3 = std::make_shared<opset8::Convert>(mul_1, element::i64);  // sizes

        opset8::Interpolate::InterpolateAttrs attrs;
        attrs.mode = opset8::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SIZES;
        attrs.nearest_mode = opset8::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        auto interpolate = std::make_shared<opset8::Interpolate>(input_1, convert_3, concat, attrs);

        auto const_4 = opset8::Constant::create(element::f32, Shape{}, {0.1f});
        auto add_1 = std::make_shared<ov::opset5::Add>(input_1, const_4);
        auto result_1 = std::make_shared<ov::opset5::Result>(add_1);
        auto result_2 = std::make_shared<ov::opset5::Result>(interpolate);

        ov::disable_fp16_compression(gather_1);
        ov::disable_fp16_compression(convert_1);
        ov::disable_fp16_compression(convert_2);
        ov::disable_fp16_compression(const_2);
        ov::disable_fp16_compression(div_1);
        ov::disable_fp16_compression(const_3);
        ov::disable_fp16_compression(concat);
        ov::disable_fp16_compression(mul_1);
        ov::disable_fp16_compression(convert_3);

        model_ref = std::make_shared<ov::Model>(NodeVector{result_1, result_2}, ParameterVector{input_1, input_2});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model_ref, model);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, MarkConstantsInShapeSubgraphs_only_consts_marked_1) {
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof = std::make_shared<opset8::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset8::Convert>(shapeof, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset8::Convert>(div, element::i64);
        auto reshape = std::make_shared<opset8::Reshape>(input_1, new_shape, false);

        model = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_2);

        auto convert_to_float = std::make_shared<opset8::Convert>(shapeof_1, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset8::Convert>(div, element::i64);
        auto reshape = std::make_shared<opset8::Reshape>(input_1, new_shape, false);

        ov::disable_fp16_compression(const_denominator);

        model_ref = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES)
                        .enable(FunctionsComparator::RUNTIME_KEYS);
    const auto res = fc.compare(model_ref, model);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, MarkConstantsInShapeSubgraphs_only_consts_marked_2) {
    // check that only constants are marked
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_1);
        auto indices = opset8::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset8::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset8::Convert>(gather_1, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset8::Convert>(div, element::i64);

        auto const_ends = opset8::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset8::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset8::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset8::Result>(slice);
        model = std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{input_1});

        pass::Manager manager;
        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 1, 720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_1);
        auto indices = opset8::Constant::create(element::i64, Shape{1}, {3});
        auto gather_axis = opset8::Constant::create(element::i64, Shape{}, {0});
        auto gather_1 = std::make_shared<opset8::Gather>(shapeof_1, indices, gather_axis);

        auto convert_to_float = std::make_shared<opset8::Convert>(gather_1, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_dim_size = std::make_shared<opset8::Convert>(div, element::i64);

        auto const_ends = opset8::Constant::create(element::i64, Shape{3}, {-1, -1, -1});
        auto concat_with_ends = std::make_shared<opset8::Concat>(OutputVector{const_ends, new_dim_size}, 0);  // scales

        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto slice = std::make_shared<opset8::StridedSlice>(input_1, begin, concat_with_ends, begin_mask, end_mask);
        auto result = std::make_shared<opset8::Result>(slice);

        ov::disable_fp16_compression(const_denominator);
        ov::disable_fp16_compression(const_ends);
        model_ref = std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{input_1});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model_ref, model);
    ASSERT_TRUE(res.valid) << res.message;
}
