// Copyright (C) 2018-2023 Intel Corporation
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
