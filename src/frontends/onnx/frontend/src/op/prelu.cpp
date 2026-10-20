// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/broadcast.hpp"
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

    // ONNX requires unidirectional trailing-axis broadcasting. Broadcasting explicitly to the data shape both
    // avoids v0::PRelu's rank-1 channel interpretation and rejects slopes that would enlarge the output.
    // Broadcast checks compatibility even when the input shapes are only known at evaluation time.
    const auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
    const auto broadcast_slope = std::make_shared<v3::Broadcast>(slope, data_shape, BroadcastType::NUMPY);
    return {std::make_shared<v0::PRelu>(data, broadcast_slope)};
}

ONNX_OP("PRelu", OPSET_SINCE(7), ai_onnx::opset_7::prelu);
}  // namespace opset_7
}  // namespace ov::frontend::onnx::ai_onnx
