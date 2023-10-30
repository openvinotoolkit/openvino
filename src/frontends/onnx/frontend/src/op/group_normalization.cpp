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
    const auto data = node.get_ng_inputs().at(0);  // Shape [N, C, ...]
    auto scale = node.get_ng_inputs().at(1);       // Shape [num_groups]
    auto bias = node.get_ng_inputs().at(2);        // Shape [num_groups]

    auto eps = node.get_attribute_value<float>("epsilon", 1e-05f);
    auto num_groups = node.get_attribute_value<int64_t>("num_groups");

    auto zero = default_opset::Constant::create(element::i64, Shape{1}, {0});
    auto one = default_opset::Constant::create(element::i64, Shape{1}, {1});
    auto c_dim = std::make_shared<default_opset::Gather>(std::make_shared<default_opset::ShapeOf>(data), one, zero);
    auto g_dim = default_opset::Constant::create(element::i64, Shape{1}, {num_groups});

    auto c_g_div = std::make_shared<default_opset::Divide>(c_dim, g_dim);

    // Adjust scale and bias shape, [G] -> [G, C/G] -> [C]
    scale = std::make_shared<default_opset::Unsqueeze>(scale, one);
    auto broadcast_scale =
        std::make_shared<default_opset::Broadcast>(scale, c_g_div, ov::op::BroadcastType::BIDIRECTIONAL);
    auto c_scale = std::make_shared<default_opset::Reshape>(broadcast_scale, c_dim, false);

    bias = std::make_shared<default_opset::Unsqueeze>(bias, one);
    auto broadcast_bias =
        std::make_shared<default_opset::Broadcast>(bias, c_g_div, ov::op::BroadcastType::BIDIRECTIONAL);
    auto c_bias = std::make_shared<default_opset::Reshape>(broadcast_bias, c_dim, false);

    return {std::make_shared<default_opset::GroupNormalization>(data, c_scale, c_bias, num_groups, eps)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
