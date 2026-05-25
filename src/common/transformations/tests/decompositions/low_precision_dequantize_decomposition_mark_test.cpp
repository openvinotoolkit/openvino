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
#include "openvino/core/model.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/rt_info/dequantization_node.hpp"

using namespace testing;

namespace {

bool find_marked_dequantization_op(const std::shared_ptr<ov::Model>& model, const ov::NodeTypeInfo& type) {
    for (const auto& op : model->get_ordered_ops()) {
        if (op->get_type_info() == type && ov::is_dequantization_node(op)) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST(DecompositionLowPrecisionDequantize, RecognizedByMarkDequantization_Asymmetric) {
    const ov::Shape weights_shape{4, 16};

    auto weights = ov::op::v0::Constant::create(ov::element::u8, weights_shape, {1});
    auto zero_point = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {127});
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.2f});

    ov::pass::NodeRegistry reg;
    auto out = ov::decomposition::low_precision_dequantize(reg, weights, scale, zero_point);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MarkDequantization>(
        ov::element::TypeVector{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
    manager.run_passes(model);

    EXPECT_TRUE(find_marked_dequantization_op(model, ov::op::v1::Subtract::get_type_info_static()))
        << "MarkDequantization did not mark the Subtract emitted by low_precision_dequantize";
    EXPECT_TRUE(find_marked_dequantization_op(model, ov::op::v1::Multiply::get_type_info_static()))
        << "MarkDequantization did not mark the Multiply emitted by low_precision_dequantize";
}

TEST(DecompositionLowPrecisionDequantize, RecognizedByMarkDequantization_Symmetric) {
    const ov::Shape weights_shape{4, 16};

    auto weights = ov::op::v0::Constant::create(ov::element::i8, weights_shape, {-2});
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.2f});

    ov::pass::NodeRegistry reg;
    auto out = ov::decomposition::low_precision_dequantize(reg, weights, scale);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MarkDequantization>(
        ov::element::TypeVector{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
    manager.run_passes(model);

    EXPECT_TRUE(find_marked_dequantization_op(model, ov::op::v1::Multiply::get_type_info_static()))
        << "MarkDequantization did not mark the Multiply emitted by low_precision_dequantize "
           "in the symmetric (no zero_point) branch";
}

TEST(DecompositionLowPrecisionDequantize, RecognizedByMarkDequantization_U4Asymmetric) {
    // Group-quantised u4 weights — the AWQ/GPTQ shape used by the PyTorch frontend.
    const ov::Shape weights_shape{2, 8, 16};
    const ov::Shape scale_shape{2, 1, 16};

    auto weights = ov::op::v0::Constant::create(ov::element::u4, weights_shape, {0});
    auto zero_point = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{}, {8});
    auto scale = ov::op::v0::Constant::create(ov::element::f16, scale_shape, {0.1f});

    ov::pass::NodeRegistry reg;
    auto out = ov::decomposition::low_precision_dequantize(reg, weights, scale, zero_point);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MarkDequantization>(
        ov::element::TypeVector{ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4});
    manager.run_passes(model);

    EXPECT_TRUE(find_marked_dequantization_op(model, ov::op::v1::Subtract::get_type_info_static()))
        << "MarkDequantization did not mark the Subtract for the u4 grouped case";
    EXPECT_TRUE(find_marked_dequantization_op(model, ov::op::v1::Multiply::get_type_info_static()))
        << "MarkDequantization did not mark the Multiply for the u4 grouped case";
}
