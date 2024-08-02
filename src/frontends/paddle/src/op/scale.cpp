// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs scale(const NodeContext& node) {
    auto data = node.get_input("X");
    auto dtype = data.get_element_type();
    // Note: paddle Scale output data_type is the same with input
    Output<Node> scale;
    Output<Node> bias;

    if (node.has_input("ScaleTensor")) {
        auto scale_tensor_node = node.get_input("ScaleTensor");
        if (scale_tensor_node.get_element_type() == dtype)
            scale = scale_tensor_node;
        else
            scale = std::make_shared<opset6::Convert>(scale_tensor_node, dtype);
    } else {
        auto scale_val = node.get_attribute<float>("scale");
        scale = ov::opset6::Constant::create(dtype, Shape{}, {scale_val});
    }

    auto bias_val = node.get_attribute<float>("bias");
    bias = ov::opset6::Constant::create(dtype, Shape{}, {bias_val});
    auto bias_after_scale = node.get_attribute<bool>("bias_after_scale");

    std::shared_ptr<Node> result_node;
    if (!bias_after_scale) {
        auto node_add = std::make_shared<ov::opset6::Add>(data, bias);
        result_node = std::make_shared<ov::opset6::Multiply>(node_add, scale);
    } else {
        auto node_multiply = std::make_shared<ov::opset6::Multiply>(data, scale);
        result_node = std::make_shared<ov::opset6::Add>(node_multiply, bias);
    }

    return node.default_single_output_mapping({result_node}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
