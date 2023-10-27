// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/group_normalization.hpp"

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector group_normalization(const Node& node) {
    const auto inputs = node.get_ng_inputs();
    OPENVINO_ASSERT(inputs.size() == 3);

    const auto& data = inputs[0];   // Shape [N, C, ...]
    const auto& scale = inputs[1];  // Shape [num_groups]
    const auto& bias = inputs[2];   // Shape [num_groups]

    const auto eps = node.get_attribute_value<float>("epsilon", 1e-05f);
    const auto num_groups = node.get_attribute_value<int64_t>("num_groups");

    const auto zero = default_opset::Constant::create(element::i64, Shape{1}, {0});
    const auto one = default_opset::Constant::create(element::i64, Shape{1}, {1});
    const auto c_dim =
        std::make_shared<default_opset::Gather>(std::make_shared<default_opset::ShapeOf>(data), one, zero);
    const auto g_dim = default_opset::Constant::create(element::i64, Shape{1}, {num_groups});

    const auto c_g_div = std::make_shared<default_opset::Divide>(c_dim, g_dim);

    // Adjust scale and bias shape, [G] -> [G, C/G] -> [C]
    const auto scale_usnsq = std::make_shared<default_opset::Unsqueeze>(scale, one);
    const auto broadcast_scale =
        std::make_shared<default_opset::Broadcast>(scale_usnsq, c_g_div, ov::op::BroadcastType::BIDIRECTIONAL);
    const auto c_scale = std::make_shared<default_opset::Reshape>(broadcast_scale, c_dim, false);

    const auto bias_unsq = std::make_shared<default_opset::Unsqueeze>(bias, one);
    const auto broadcast_bias =
        std::make_shared<default_opset::Broadcast>(bias_unsq, c_g_div, ov::op::BroadcastType::BIDIRECTIONAL);
    const auto c_bias = std::make_shared<default_opset::Reshape>(broadcast_bias, c_dim, false);

    return {std::make_shared<default_opset::GroupNormalization>(data, c_scale, c_bias, num_groups, eps)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
