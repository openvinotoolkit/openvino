// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
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

    const auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
    const auto data_rank = data.get_partial_shape().rank();
    const auto slope_rank = slope.get_partial_shape().rank();
    if (data_rank.is_static() && slope_rank.is_static() && slope_rank.get_length() <= data_rank.get_length()) {
        // Validate against the trailing data dimensions without expanding singleton slope dimensions.
        // Keeping this check in the graph also rejects incompatible shapes after specialization or at runtime.
        std::vector<int64_t> trailing_axes;
        for (int64_t axis = -slope_rank.get_length(); axis < 0; ++axis) {
            trailing_axes.push_back(axis);
        }
        const auto axes = v0::Constant::create(ov::element::i64, ov::Shape{trailing_axes.size()}, trailing_axes);
        const auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        const auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        const auto trailing_shape = std::make_shared<v8::Gather>(data_shape, axes, zero);
        const auto slope_shape = std::make_shared<v3::ShapeOf>(slope, ov::element::i64);
        const auto singleton_dims = std::make_shared<v1::Equal>(slope_shape, one);
        const auto compact_target = std::make_shared<v1::Select>(singleton_dims, slope_shape, trailing_shape);
        ov::Output<ov::Node> checked_slope =
            std::make_shared<v3::Broadcast>(slope, compact_target, BroadcastType::NUMPY);
        if (slope_rank.get_length() == 1 && data_rank.get_length() >= 2) {
            // Only rank-1 slopes need reshaping to avoid PRelu's channel-axis interpretation.
            std::vector<int64_t> pattern(data_rank.get_length(), 1);
            pattern.back() = -1;
            const auto target = v0::Constant::create(ov::element::i64, ov::Shape{pattern.size()}, pattern);
            checked_slope = std::make_shared<v1::Reshape>(checked_slope, target, false);
        }
        return {std::make_shared<v0::PRelu>(data, checked_slope)};
    }

    // Unknown ranks need a checked fallback: never allow the slope to enlarge PRelu's output.
    const auto broadcast_slope = std::make_shared<v3::Broadcast>(slope, data_shape, BroadcastType::NUMPY);
    return {std::make_shared<v0::PRelu>(data, broadcast_slope)};
}

ONNX_OP("PRelu", OPSET_SINCE(7), ai_onnx::opset_7::prelu);
}  // namespace opset_7
}  // namespace ov::frontend::onnx::ai_onnx
