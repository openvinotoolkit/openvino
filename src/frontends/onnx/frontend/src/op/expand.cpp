// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector expand(const ov::frontend::onnx::Node& node) {
    const ov::Output<ov::Node> data{node.get_ov_inputs().at(0)};
    const ov::Output<ov::Node> shape{node.get_ov_inputs().at(1)};

    if (common::is_failsafe_node(shape.get_node_shared_ptr()) ||
        common::is_constant_empty_node(shape.get_node_shared_ptr())) {
        // Ignore an unusable target shape, such as a failsafe node created for an invalid
        // initializer or an empty constant. Use an identity broadcast so Expand preserves
        // the input tensor, and let a later optimization pass eliminate this Broadcast.
        const auto identity_broadcast = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        return {std::make_shared<v3::Broadcast>(data, identity_broadcast, ov::op::BroadcastType::BIDIRECTIONAL)};
    } else {
        return {std::make_shared<v3::Broadcast>(data, shape, ov::op::BroadcastType::BIDIRECTIONAL)};
    }
}

ONNX_OP("Expand", OPSET_SINCE(1), ai_onnx::opset_1::expand);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
