// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx {
namespace opset_1 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto& data = ov_inputs.at(0);
    const auto& slope = ov_inputs.at(1);

    // PRelu-1/6 use the legacy (Caffe2-like) broadcasting: a slope of size 1 is shared across channels, i.e. a
    // rank-1 slope is aligned with the channel axis. OpenVINO's v0::PRelu already implements exactly this behavior,
    // so the inputs are forwarded without any reshaping.
    return {std::make_shared<v0::PRelu>(data, slope)};
}

ONNX_OP("PRelu", OPSET_RANGE(1, 6), ai_onnx::opset_1::prelu);
}  // namespace opset_1

namespace opset_7 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto& data = ov_inputs.at(0);
    const auto& slope = ov_inputs.at(1);

    // v0::PRelu only reinterprets a rank-1 slope as per-channel; any other static rank already broadcasts with
    // ONNX (NUMPY, trailing-axis) rules, so forward it without the normalization subgraph.
    const auto& static_slope_rank = slope.get_partial_shape().rank();
    if (static_slope_rank.is_static() && static_slope_rank.get_length() != 1) {
        return {std::make_shared<v0::PRelu>(data, slope)};
    }

    const auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
    const auto slope_shape = std::make_shared<v3::ShapeOf>(slope, ov::element::i64);
    const auto data_rank_vector = std::make_shared<v3::ShapeOf>(data_shape, ov::element::i64);
    const auto slope_rank_vector = std::make_shared<v3::ShapeOf>(slope_shape, ov::element::i64);
    const auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto data_rank = std::make_shared<v0::Squeeze>(data_rank_vector);
    const auto slope_rank = std::make_shared<v0::Squeeze>(slope_rank_vector);
    const auto axes_start = std::make_shared<v1::Subtract>(data_rank, slope_rank);
    const auto trailing_axes = std::make_shared<v4::Range>(axes_start, data_rank, one, ov::element::i64);
    const auto trailing_shape = std::make_shared<v8::Gather>(data_shape, trailing_axes, zero);
    const auto singleton_dims = std::make_shared<v1::Equal>(slope_shape, one);
    const auto compact_target = std::make_shared<v1::Select>(singleton_dims, slope_shape, trailing_shape);
    const auto leading_ones =
        std::make_shared<v3::Broadcast>(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                        std::make_shared<v1::Subtract>(data_rank_vector, slope_rank_vector));
    const auto normalized_shape = std::make_shared<v0::Concat>(ov::OutputVector{leading_ones, compact_target}, 0);

    // Everything is derived from the current shapes so a post-conversion rank change keeps non-singleton slope
    // dimensions on the trailing data axes. Broadcasting to this compact target left-pads the slope and rejects
    // dimensions that are neither 1 nor equal to the data, without materializing an activation-sized slope.
    const auto normalized_slope = std::make_shared<v3::Broadcast>(slope, normalized_shape, BroadcastType::NUMPY);
    return {std::make_shared<v0::PRelu>(data, normalized_slope)};
}

ONNX_OP("PRelu", OPSET_SINCE(7), ai_onnx::opset_7::prelu);
}  // namespace opset_7
}  // namespace ov::frontend::onnx::ai_onnx
