// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/compress_float_constants.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, CompressConstants_f32) {
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f32,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        auto const_scales = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {1.4});

        auto shape = std::make_shared<ov::opset8::ShapeOf>(conv);
        auto convert1 = std::make_shared<ov::opset8::Convert>(shape, ov::element::f32);
        auto mul = std::make_shared<ov::opset8::Multiply>(convert1, const_scales);
        auto convert2 = std::make_shared<ov::opset8::Convert>(mul, ov::element::i32);

        auto default_scales_node = ov::opset8::Constant::create(ov::element::f32, ov::Shape{4}, {1., 1., 1.4, 1.4});
        auto axes_node = ov::opset8::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});

        auto interpolate4_attr =
            ov::opset8::Interpolate::InterpolateAttrs(ov::opset8::Interpolate::InterpolateMode::NEAREST,
                                                      ov::opset8::Interpolate::ShapeCalcMode::SIZES,
                                                      std::vector<size_t>{0, 0, 0, 0},
                                                      std::vector<size_t>{0, 0, 0, 0},
                                                      ov::opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                                                      ov::opset8::Interpolate::NearestMode::SIMPLE,
                                                      false,
                                                      -0.75);

        auto resize = std::make_shared<ov::opset8::Interpolate>(conv,
                                                                convert2,
                                                                default_scales_node,
                                                                axes_node,
                                                                interpolate4_attr);

        model = std::make_shared<ov::Model>(ov::NodeVector{resize}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        auto const_scales = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {1.4});

        auto shape = std::make_shared<ov::opset8::ShapeOf>(conv);
        auto convert1 = std::make_shared<ov::opset8::Convert>(shape, ov::element::f32);
        auto mul = std::make_shared<ov::opset8::Multiply>(convert1, const_scales);
        auto convert2 = std::make_shared<ov::opset8::Convert>(mul, ov::element::i32);

        auto default_scales_node = ov::opset8::Constant::create(ov::element::f32, ov::Shape{4}, {1., 1., 1.4, 1.4});
        auto axes_node = ov::opset8::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});

        auto interpolate4_attr =
            ov::opset8::Interpolate::InterpolateAttrs(ov::opset8::Interpolate::InterpolateMode::NEAREST,
                                                      ov::opset8::Interpolate::ShapeCalcMode::SIZES,
                                                      std::vector<size_t>{0, 0, 0, 0},
                                                      std::vector<size_t>{0, 0, 0, 0},
                                                      ov::opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                                                      ov::opset8::Interpolate::NearestMode::SIMPLE,
                                                      false,
                                                      -0.75);

        auto resize = std::make_shared<ov::opset8::Interpolate>(conv,
                                                                convert2,
                                                                default_scales_node,
                                                                axes_node,
                                                                interpolate4_attr);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{resize}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, CompressConstants_f32_If) {
    {
        // create then body
        auto input_then = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f32,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input_then,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        auto const_scales = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {1.4});

        auto shape = std::make_shared<ov::opset8::ShapeOf>(conv);
        auto convert1 = std::make_shared<ov::opset8::Convert>(shape, ov::element::f32);
        auto mul = std::make_shared<ov::opset8::Multiply>(convert1, const_scales);
        auto convert2 = std::make_shared<ov::opset8::Convert>(mul, ov::element::i32);

        auto default_scales_node = ov::opset8::Constant::create(ov::element::f32, ov::Shape{4}, {1., 1., 1.4, 1.4});
        auto axes_node = ov::opset8::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});

        auto interpolate4_attr =
            ov::opset8::Interpolate::InterpolateAttrs(ov::opset8::Interpolate::InterpolateMode::NEAREST,
                                                      ov::opset8::Interpolate::ShapeCalcMode::SIZES,
                                                      std::vector<size_t>{0, 0, 0, 0},
                                                      std::vector<size_t>{0, 0, 0, 0},
                                                      ov::opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                                                      ov::opset8::Interpolate::NearestMode::SIMPLE,
                                                      false,
                                                      -0.75);

        auto resize = std::make_shared<ov::opset8::Interpolate>(conv,
                                                                convert2,
                                                                default_scales_node,
                                                                axes_node,
                                                                interpolate4_attr);
        auto then_op_result = std::make_shared<ov::op::v0::Result>(resize);
        auto body_then_function =
            std::make_shared<ov::Model>(ov::NodeVector{then_op_result}, ov::ParameterVector{input_then});

        // create else body
        auto input_else = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto else_op_result = std::make_shared<ov::op::v0::Result>(input_else);
        auto body_else_function =
            std::make_shared<ov::Model>(ov::NodeVector{else_op_result}, ov::ParameterVector{input_else});

        // create main graph
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto cond = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
        auto if_op = std::make_shared<ov::opset8::If>(cond);
        if_op->set_then_body(body_then_function);
        if_op->set_else_body(body_else_function);
        if_op->set_input(input, input_then, input_else);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<ov::op::v0::Result>(if_op);

        model = std::make_shared<ov::Model>(NodeVector{if_result}, ParameterVector{input});

        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        // create then body
        auto input_then = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input_then,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        auto const_scales = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {1.4});

        auto shape = std::make_shared<ov::opset8::ShapeOf>(conv);
        auto convert1 = std::make_shared<ov::opset8::Convert>(shape, ov::element::f32);
        auto mul = std::make_shared<ov::opset8::Multiply>(convert1, const_scales);
        auto convert2 = std::make_shared<ov::opset8::Convert>(mul, ov::element::i32);

        auto default_scales_node = ov::opset8::Constant::create(ov::element::f32, ov::Shape{4}, {1., 1., 1.4, 1.4});
        auto axes_node = ov::opset8::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});

        auto interpolate4_attr =
            ov::opset8::Interpolate::InterpolateAttrs(ov::opset8::Interpolate::InterpolateMode::NEAREST,
                                                      ov::opset8::Interpolate::ShapeCalcMode::SIZES,
                                                      std::vector<size_t>{0, 0, 0, 0},
                                                      std::vector<size_t>{0, 0, 0, 0},
                                                      ov::opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                                                      ov::opset8::Interpolate::NearestMode::SIMPLE,
                                                      false,
                                                      -0.75);

        auto resize = std::make_shared<ov::opset8::Interpolate>(conv,
                                                                convert2,
                                                                default_scales_node,
                                                                axes_node,
                                                                interpolate4_attr);
        auto then_op_result = std::make_shared<ov::op::v0::Result>(resize);
        auto body_then_function =
            std::make_shared<ov::Model>(ov::NodeVector{then_op_result}, ov::ParameterVector{input_then});

        // create else body
        auto input_else = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto else_op_result = std::make_shared<ov::op::v0::Result>(input_else);
        auto body_else_function =
            std::make_shared<ov::Model>(ov::NodeVector{else_op_result}, ov::ParameterVector{input_else});

        // create main graph
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto cond = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
        auto if_op = std::make_shared<ov::opset8::If>(cond);
        if_op->set_then_body(body_then_function);
        if_op->set_else_body(body_else_function);
        if_op->set_input(input, input_then, input_else);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<ov::op::v0::Result>(if_op);

        model_ref = std::make_shared<ov::Model>(NodeVector{if_result}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, CompressConstants_f64) {
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f64, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f64,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f64, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f64);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, CompressConstants_keep_in_f32_small_eps_out_of_range) {
    float fp16_eps = 0.00000001f;  // smaller than fp16 minimal value: float16::from_bits(0x0001)
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        // fp16_eps is lesser that fp16 minimal value,
        // they must be stored in fp32 because of a big proportion of such out of range values
        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps,
                                                           fp16_eps});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, CompressConstants_keep_in_f32_max_out_of_range_val) {
    // if fp16 out of range values fraction is greater than threshold (75%) then keep them in fp32
    // no decompression converts should be inserted
    float fp16_oor = static_cast<float>(std::numeric_limits<ov::float16>::max()) + 100.0f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
                                                          ov::Shape{1, 3, 4, 1},
                                                          {0.0f,
                                                           1.0f,
                                                           2.0f,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor,
                                                           fp16_oor});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, CompressConstants_compress_to_f16_max_out_of_range_val) {
    // fp16 out of range should be clipped to fp16_max_val if fraction of out of range values is less than threshold
    float fp16_oor = static_cast<float>(std::numeric_limits<ov::float16>::max()) + 100.0f;
    float fp16_max = static_cast<float>(std::numeric_limits<ov::float16>::max());
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        // only half of values are out of range, therefore they will be compressed to fp16
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f32,
            ov::Shape{1, 3, 4, 1},
            {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, fp16_oor, fp16_oor, fp16_oor, fp16_oor, fp16_oor, fp16_oor});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 4, 1},
            {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, fp16_max, fp16_max, fp16_max, fp16_max, fp16_max, fp16_max});
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, CompressConstants_not_keep_in_f32_when_zeros) {
    // zero values are less than fp16_eps, but they are exactly expressed in fp16
    // not need to keep them in fp32
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        auto const_weights =
            ov::opset8::Constant::create(ov::element::f32,
                                         ov::Shape{1, 3, 4, 1},
                                         {0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights =
            ov::opset8::Constant::create(ov::element::f16,
                                         ov::Shape{1, 3, 4, 1},
                                         {0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, CompressConstants_compress_to_f16_denormal_vals) {
    float fp16_denormal = 0.00001f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});

        // only one third of values are out of fp16 normal range, therefore they will be compressed to fp16
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f32,
            ov::Shape{1, 3, 3, 1},
            {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, fp16_denormal, fp16_denormal, fp16_denormal});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 1},
            {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, fp16_denormal, fp16_denormal, fp16_denormal});
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, KeepFWPrecisionForFP16Constants_test_1) {
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::op::v0::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_node = std::make_shared<ov::op::v0::Convert>(const_weights, element::f32);

        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_node,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        auto convert_node = std::make_shared<ov::op::v0::Convert>(const_weights, element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_node,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, KeepFWPrecisionForBF16Constants_test_1) {
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::op::v0::Constant::create(
            ov::element::bf16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_node = std::make_shared<ov::op::v0::Convert>(const_weights, element::f32);

        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_node,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::bf16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

        auto convert_node = std::make_shared<ov::op::v0::Convert>(const_weights, element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_node,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

namespace {
struct TestParams {
    TestParams() = default;
    bool has_subtract = {};
    ov::element::Type element_type = {};
};

auto build_model_DetectFakeQuantize = [](const TestParams&) -> std::shared_ptr<ov::Model> {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto input_low = ov::op::v0::Constant::create(element::u8, Shape{}, {1});
    auto input_high = ov::op::v0::Constant::create(element::u8, Shape{}, {2});
    auto output_low = ov::op::v0::Constant::create(element::u8, Shape{}, {1});
    auto output_high = ov::op::v0::Constant::create(element::u8, Shape{}, {2});
    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 1);
    return std::make_shared<ov::Model>(ov::NodeVector{fq}, ov::ParameterVector{input});
};

auto build_model_DetectFakeConvert = [](const TestParams&) -> std::shared_ptr<ov::Model> {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto scale = ov::op::v0::Constant::create(element::f32, Shape{}, {1});
    auto convert = std::make_shared<ov::op::v13::FakeConvert>(input, scale);
    return std::make_shared<ov::Model>(ov::NodeVector{convert}, ov::ParameterVector{input});
};

auto build_model_DetectCompressedWeights = [](const TestParams& params) -> std::shared_ptr<ov::Model> {
    auto input = std::make_shared<op::v0::Parameter>(element::u8, Shape{2, 1});

    auto weights = ov::op::v0::Constant::create(params.element_type, ov::Shape{1, 2}, {2});
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, element::u8);

    std::shared_ptr<Node> tail_node = convert;
    if (params.has_subtract) {
        auto zero_point_const = ov::op::v0::Constant::create(element::u8, ov::Shape{1, 2}, {2});
        auto zero_point = std::make_shared<ov::op::v0::Convert>(weights, element::u8);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zero_point);
        tail_node = subtract;
    }
    auto multiply_const = ov::op::v0::Constant::create(element::u8, ov::Shape{1, 2}, {2});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(tail_node, multiply_const);

    auto out_multiply = std::make_shared<ov::op::v1::Multiply>(input, multiply);
    return std::make_shared<ov::Model>(ov::NodeVector{out_multiply}, ov::ParameterVector{input});
};

using ModelFactoryFunc = std::function<std::shared_ptr<Model>(const TestParams&)>;

struct ModelFactory {
    ModelFactory(ModelFactoryFunc func, std::string func_name) : create(std::move(func)), name(std::move(func_name)) {}
    ModelFactoryFunc create;
    std::string name;
};

using CheckModelOptimizedParam = std::tuple<ModelFactory,        // model factory
                                            bool,                // has_subtract
                                            ov::element::Type>;  // element_type

const std::vector<ov::element::Type> element_types = {ov::element::i4,
                                                      ov::element::u4,
                                                      ov::element::i8,
                                                      ov::element::u8,
                                                      ov::element::nf4,
                                                      ov::element::f8e4m3,
                                                      ov::element::f8e5m2};
}  // namespace

class CheckModelOptimizedTestSuite : public testing::TestWithParam<CheckModelOptimizedParam> {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<CheckModelOptimizedParam>& obj) {
        auto model_factory = std::get<0>(obj.param);
        TestParams params;
        params.has_subtract = std::get<1>(obj.param);
        params.element_type = std::get<2>(obj.param);

        std::ostringstream test_name;
        test_name << "model_factory=" << model_factory.name << "/";
        test_name << "has_subtract=" << params.has_subtract << "/";
        test_name << "element_type=" << params.element_type;

        return test_name.str();
    }
};

TEST_P(CheckModelOptimizedTestSuite, CheckModelOptimized) {
    const auto& param = GetParam();
    auto model_factory = std::get<0>(param);
    TestParams params;
    params.has_subtract = std::get<1>(param);
    params.element_type = std::get<2>(param);
    auto model = model_factory.create(params);

    ASSERT_TRUE(ov::pass::is_model_optimized(model));
}

#undef ADD_FACTORY
#define ADD_FACTORY(function) ModelFactory(function, #function)

INSTANTIATE_TEST_SUITE_P(CheckModelOptimizedFakeQuantizeConvert,
                         CheckModelOptimizedTestSuite,
                         testing::Combine(testing::Values(ADD_FACTORY(build_model_DetectFakeQuantize),
                                                          ADD_FACTORY(build_model_DetectFakeConvert)),
                                          testing::Values(false),
                                          testing::Values(ov::element::Type())),
                         CheckModelOptimizedTestSuite::get_test_name);

INSTANTIATE_TEST_SUITE_P(CheckModelOptimizedDetectCompressedWeights,
                         CheckModelOptimizedTestSuite,
                         testing::Combine(testing::Values(ADD_FACTORY(build_model_DetectCompressedWeights)),
                                          testing::Values(true, false),
                                          testing::ValuesIn(element_types)),
                         CheckModelOptimizedTestSuite::get_test_name);
