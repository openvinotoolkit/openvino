// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests that the canonical decomposition emitted by
// ov::decomposition::low_precision_dequantize is recognised by
// ov::pass::MarkDequantization. These tests act as a guard for both
// directions:
//   * decomposition authors must keep the produced sub-graph in the shape
//     accepted by MarkDequantization;
//   * MarkDequantization authors must keep accepting the canonical
//     decomposition shape that frontends emit.

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

using namespace ov;

namespace {

const element::TypeVector kLowPrecisionTypes{element::u8, element::i8, element::u4, element::i4};

}  // namespace

TEST_F(TransformationTestsF, LowPrecisionDequantize_Asymmetric) {
    const Shape weights_shape{4, 16};

    {
        auto weights = op::v0::Constant::create(element::u8, weights_shape, {1});
        auto zero_point = op::v0::Constant::create(element::u8, Shape{}, {127});
        auto scale = op::v0::Constant::create(element::f32, Shape{}, {0.2f});

        auto out = decomposition::low_precision_dequantize(weights, scale, zero_point);
        model = std::make_shared<Model>(OutputVector{out}, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(kLowPrecisionTypes);
    manager.register_pass<pass::ConstantFolding>();

    {
        auto weights = op::v0::Constant::create(element::u8, weights_shape, {1});
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        disable_constant_folding(convert);

        auto zero_point = op::v0::Constant::create(element::u8, Shape{}, {127});
        auto zp_convert = std::make_shared<op::v0::Convert>(zero_point, element::f32);
        disable_constant_folding(zp_convert);

        auto subtract = std::make_shared<op::v1::Subtract>(convert, zp_convert);
        mark_as_dequantization_node(subtract);

        auto scale = op::v0::Constant::create(element::f32, Shape{}, {0.2f});
        auto multiply = std::make_shared<op::v1::Multiply>(subtract, scale);
        mark_as_dequantization_node(multiply);

        model_ref = std::make_shared<Model>(OutputVector{multiply}, ParameterVector{});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, LowPrecisionDequantize_Symmetric) {
    const Shape weights_shape{4, 16};

    {
        auto weights = op::v0::Constant::create(element::i8, weights_shape, {-2});
        auto scale = op::v0::Constant::create(element::f32, Shape{}, {0.2f});

        auto out = decomposition::low_precision_dequantize(weights, scale);
        model = std::make_shared<Model>(OutputVector{out}, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(kLowPrecisionTypes);
    manager.register_pass<pass::ConstantFolding>();

    {
        auto weights = op::v0::Constant::create(element::i8, weights_shape, {-2});
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        disable_constant_folding(convert);

        auto scale = op::v0::Constant::create(element::f32, Shape{}, {0.2f});
        auto multiply = std::make_shared<op::v1::Multiply>(convert, scale);
        mark_as_dequantization_node(multiply);

        model_ref = std::make_shared<Model>(OutputVector{multiply}, ParameterVector{});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

TEST_F(TransformationTestsF, LowPrecisionDequantize_U4Asymmetric) {
    // Group-quantised u4 weights — the AWQ/GPTQ shape used by the PyTorch frontend.
    const Shape weights_shape{2, 8, 16};
    const Shape scale_shape{2, 1, 16};

    {
        auto weights = op::v0::Constant::create(element::u4, weights_shape, {0});
        auto zero_point = op::v0::Constant::create(element::u4, Shape{}, {8});
        auto scale = op::v0::Constant::create(element::f16, scale_shape, {0.1f});

        auto out = decomposition::low_precision_dequantize(weights, scale, zero_point);
        model = std::make_shared<Model>(OutputVector{out}, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(kLowPrecisionTypes);
    manager.register_pass<pass::ConstantFolding>();

    {
        auto weights = op::v0::Constant::create(element::u4, weights_shape, {0});
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f16);
        disable_constant_folding(convert);

        auto zero_point = op::v0::Constant::create(element::u4, Shape{}, {8});
        auto zp_convert = std::make_shared<op::v0::Convert>(zero_point, element::f16);
        disable_constant_folding(zp_convert);

        auto subtract = std::make_shared<op::v1::Subtract>(convert, zp_convert);
        mark_as_dequantization_node(subtract);

        auto scale = op::v0::Constant::create(element::f16, scale_shape, {0.1f});
        auto multiply = std::make_shared<op::v1::Multiply>(subtract, scale);
        mark_as_dequantization_node(multiply);

        model_ref = std::make_shared<Model>(OutputVector{multiply}, ParameterVector{});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

// Reshape branch — explicit output_shape that differs from the Multiply output.
// ONNX blocked DequantizeLinear reshapes a [num_blocks, block_size, ...] grid
// back to the input rank.
TEST_F(TransformationTestsF, LowPrecisionDequantize_WithReshape) {
    const Shape grid_shape{2, 4, 16};  // num_blocks, block_size, channels
    const Shape scale_shape{2, 1, 16};
    const std::vector<int32_t> target_shape{8, 16};  // num_blocks * block_size, channels

    {
        auto weights = op::v0::Constant::create(element::u8, grid_shape, {0});
        auto zero_point = op::v0::Constant::create(element::u8, Shape{}, {8});
        auto scale = op::v0::Constant::create(element::f32, scale_shape, {0.1f});
        auto out_shape = op::v0::Constant::create(element::i32, Shape{2}, target_shape);

        auto out = decomposition::low_precision_dequantize(weights, scale, zero_point, out_shape);
        model = std::make_shared<Model>(OutputVector{out}, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(kLowPrecisionTypes);
    manager.register_pass<pass::ConstantFolding>();

    {
        auto weights = op::v0::Constant::create(element::u8, grid_shape, {0});
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        disable_constant_folding(convert);

        auto zero_point = op::v0::Constant::create(element::u8, Shape{}, {8});
        auto zp_convert = std::make_shared<op::v0::Convert>(zero_point, element::f32);
        disable_constant_folding(zp_convert);

        auto subtract = std::make_shared<op::v1::Subtract>(convert, zp_convert);
        mark_as_dequantization_node(subtract);

        auto scale = op::v0::Constant::create(element::f32, scale_shape, {0.1f});
        auto multiply = std::make_shared<op::v1::Multiply>(subtract, scale);
        mark_as_dequantization_node(multiply);

        auto out_shape = op::v0::Constant::create(element::i32, Shape{2}, target_shape);
        auto reshape = std::make_shared<op::v1::Reshape>(multiply, out_shape, /*special_zero=*/false);

        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

// When output_shape matches the Multiply output shape, the helper should skip
// the trailing Reshape entirely.
TEST_F(TransformationTestsF, LowPrecisionDequantize_NoopReshapeSkipped) {
    const Shape weights_shape{4, 16};
    const std::vector<int32_t> target_shape{4, 16};

    {
        auto weights = op::v0::Constant::create(element::u8, weights_shape, {1});
        auto zero_point = op::v0::Constant::create(element::u8, Shape{}, {127});
        auto scale = op::v0::Constant::create(element::f32, Shape{}, {0.2f});
        auto out_shape = op::v0::Constant::create(element::i32, Shape{2}, target_shape);

        auto out = decomposition::low_precision_dequantize(weights, scale, zero_point, out_shape);
        model = std::make_shared<Model>(OutputVector{out}, ParameterVector{});
    }

    manager.register_pass<pass::MarkDequantization>(kLowPrecisionTypes);
    manager.register_pass<pass::ConstantFolding>();

    {
        auto weights = op::v0::Constant::create(element::u8, weights_shape, {1});
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        disable_constant_folding(convert);

        auto zero_point = op::v0::Constant::create(element::u8, Shape{}, {127});
        auto zp_convert = std::make_shared<op::v0::Convert>(zero_point, element::f32);
        disable_constant_folding(zp_convert);

        auto subtract = std::make_shared<op::v1::Subtract>(convert, zp_convert);
        mark_as_dequantization_node(subtract);

        auto scale = op::v0::Constant::create(element::f32, Shape{}, {0.2f});
        auto multiply = std::make_shared<op::v1::Multiply>(subtract, scale);
        mark_as_dequantization_node(multiply);

        // No Reshape — helper detects output_shape == current shape and skips it.
        model_ref = std::make_shared<Model>(OutputVector{multiply}, ParameterVector{});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}
